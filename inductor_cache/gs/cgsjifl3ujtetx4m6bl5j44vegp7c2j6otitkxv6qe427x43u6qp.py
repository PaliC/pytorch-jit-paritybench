# AOT ID: ['29_forward']
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


# kernel path: inductor_cache/mw/cmws5xwnujklmhhyvsgjebkmxyvolsufnqyrxzc7m635xord5taf.py
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
    size_hints={'y': 64, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39
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


# kernel path: inductor_cache/gr/cgrl5qudizuxta33u254nsdpoimnwfci36h4hipspoifaljbbsuh.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
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


# kernel path: inductor_cache/gj/cgjhoybcaddq57wpn7qbdkgu6q5i5uvxr2fn7s6zysjl2yztqdvm.py
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
    size_hints={'y': 4096, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/v4/cv47h4p3i5fqre7b5aku7w3aftg2x73vgp2rzroh75thcl4i4uf7.py
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
    size_hints={'y': 16384, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 3
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
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 384*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rn/crnv6jve4uqkn3j3hudgu2a2qmkn32s7nt2fjjx2e7pvk5bfwejb.py
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
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/5d/c5dlvdgknplmdruf5owlzixvcy7cji36emgtjuz52so6p7uhkcet.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 144*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qy/cqyc4rqi4dve4ufnbxmhhect2y5qccz2uqq5czkfjyocvlh4bjds.py
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
    size_hints={'y': 256, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/6c/c6cvezd6m66fzq5ubehdhc377w3sewjmtnqwkvwmnrctyil3jce6.py
# Topologically Sorted Source Nodes: [pool_out, pool_out_1], Original ATen: [aten.max_pool2d_with_indices, aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
# Source node to ATen node mapping:
#   pool_out => _low_memory_max_pool2d_with_offsets
#   pool_out_1 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add, add_4, add_5, clamp_max_2, clamp_min, clamp_min_2, convert_element_type, convert_element_type_3, iota, mul, mul_2, mul_3, sub, sub_2, sub_3, sub_4
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%primals_2, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 1.0), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min, torch.int64), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_3), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_max_pool2d_with_indices_mul_sub_9 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_max_pool2d_with_indices_mul_sub_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_max_pool2d_with_indices_mul_sub_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_max_pool2d_with_indices_mul_sub_9(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = ((xindex // 1024) % 3)
    x3 = xindex // 3072
    x5 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = x0
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12 * tmp4
    tmp14 = tmp13 - tmp2
    tmp15 = triton_helpers.maximum(tmp14, tmp7)
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tl.full([1], 1, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1], 31, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tl.load(in_ptr0 + (x2 + 6*tmp20 + 384*tmp9 + 12288*x3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (3 + x2 + 6*tmp20 + 384*tmp9 + 12288*x3), None, eviction_policy='evict_last')
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tmp24 = tl.load(in_ptr0 + (192 + x2 + 6*tmp20 + 384*tmp9 + 12288*x3), None, eviction_policy='evict_last')
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tmp26 = tl.load(in_ptr0 + (195 + x2 + 6*tmp20 + 384*tmp9 + 12288*x3), None, eviction_policy='evict_last')
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tmp28 = tl.load(in_ptr0 + (x2 + 6*tmp16 + 384*tmp9 + 12288*x3), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (3 + x2 + 6*tmp16 + 384*tmp9 + 12288*x3), None, eviction_policy='evict_last')
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tmp31 = tl.load(in_ptr0 + (192 + x2 + 6*tmp16 + 384*tmp9 + 12288*x3), None, eviction_policy='evict_last')
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tmp33 = tl.load(in_ptr0 + (195 + x2 + 6*tmp16 + 384*tmp9 + 12288*x3), None, eviction_policy='evict_last')
    tmp34 = triton_helpers.maximum(tmp33, tmp32)
    tmp35 = tmp27 - tmp34
    tmp36 = tmp9 + tmp17
    tmp37 = triton_helpers.minimum(tmp36, tmp19)
    tmp38 = tl.load(in_ptr0 + (x2 + 6*tmp20 + 384*tmp37 + 12288*x3), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (3 + x2 + 6*tmp20 + 384*tmp37 + 12288*x3), None, eviction_policy='evict_last')
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tmp41 = tl.load(in_ptr0 + (192 + x2 + 6*tmp20 + 384*tmp37 + 12288*x3), None, eviction_policy='evict_last')
    tmp42 = triton_helpers.maximum(tmp41, tmp40)
    tmp43 = tl.load(in_ptr0 + (195 + x2 + 6*tmp20 + 384*tmp37 + 12288*x3), None, eviction_policy='evict_last')
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tmp45 = tl.load(in_ptr0 + (x2 + 6*tmp16 + 384*tmp37 + 12288*x3), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr0 + (3 + x2 + 6*tmp16 + 384*tmp37 + 12288*x3), None, eviction_policy='evict_last')
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tmp48 = tl.load(in_ptr0 + (192 + x2 + 6*tmp16 + 384*tmp37 + 12288*x3), None, eviction_policy='evict_last')
    tmp49 = triton_helpers.maximum(tmp48, tmp47)
    tmp50 = tl.load(in_ptr0 + (195 + x2 + 6*tmp16 + 384*tmp37 + 12288*x3), None, eviction_policy='evict_last')
    tmp51 = triton_helpers.maximum(tmp50, tmp49)
    tmp52 = tmp44 - tmp51
    tmp53 = tmp16.to(tl.float32)
    tmp54 = tmp15 - tmp53
    tmp55 = triton_helpers.maximum(tmp54, tmp7)
    tmp56 = triton_helpers.minimum(tmp55, tmp4)
    tmp57 = tmp35 * tmp56
    tmp58 = tmp34 + tmp57
    tmp59 = tmp52 * tmp56
    tmp60 = tmp51 + tmp59
    tl.store(in_out_ptr0 + (x5), tmp58, None)
    tl.store(in_out_ptr1 + (x5), tmp60, None)
''', device_str='cuda')


# kernel path: inductor_cache/pb/cpbubi2oejliuymqamkkz7enj75w25fcm3h7vkk6iwk65jcbvazb.py
# Topologically Sorted Source Nodes: [output, output_1, output_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output => cat
#   output_1 => add_8, mul_6, mul_7, sub_7
#   output_2 => gt, mul_8, where
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %add_6], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat, %unsqueeze_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_5), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_7), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_8, 0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %add_8), kwargs = {})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_8, %mul_8), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x4 = xindex // 16
    x3 = xindex // 16384
    x6 = ((xindex // 16) % 1024)
    x2 = ((xindex // 512) % 32)
    x5 = xindex
    tmp31 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr7 + (0))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 13, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (13*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 16, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x6 + 1024*((-13) + x0) + 3072*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x6 + 1024*((-13) + x0) + 3072*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 - tmp9
    tmp12 = x2
    tmp13 = tmp12.to(tl.float32)
    tmp14 = 0.5
    tmp15 = tmp13 + tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 - tmp14
    tmp19 = 0.0
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = tmp20.to(tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 - tmp22
    tmp24 = triton_helpers.maximum(tmp23, tmp19)
    tmp25 = triton_helpers.minimum(tmp24, tmp16)
    tmp26 = tmp11 * tmp25
    tmp27 = tmp9 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp6, tmp27, tmp28)
    tmp30 = tl.where(tmp4, tmp5, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = tl.full([1], 1, tl.int32)
    tmp38 = tmp37 / tmp36
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp32 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = 0.0
    tmp47 = tmp45 > tmp46
    tmp50 = tmp49 * tmp45
    tmp51 = tl.where(tmp47, tmp45, tmp50)
    tl.store(out_ptr0 + (x5), tmp30, None)
    tl.store(in_out_ptr0 + (x5), tmp51, None)
''', device_str='cuda')


# kernel path: inductor_cache/57/c574uk2avnmbda63jx2r2x2czynuqjcyjuiqnatxfzcpazyfcxao.py
# Topologically Sorted Source Nodes: [pool_out_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   pool_out_2 => getitem_3
# Graph fragment:
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_11 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x1 + 1024*x2), None)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 32*x1 + 1024*x2), None)
    tmp7 = tl.load(in_ptr0 + (512 + x0 + 32*x1 + 1024*x2), None)
    tmp12 = tl.load(in_ptr0 + (528 + x0 + 32*x1 + 1024*x2), None)
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


# kernel path: inductor_cache/z5/cz5b3xb52chfd6wz66dophbzzoaqm2rfidxxvzcvfnim6csokoy7.py
# Topologically Sorted Source Nodes: [pool_out_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   pool_out_3 => convert_element_type_7
# Graph fragment:
#   %convert_element_type_7 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.int64), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yq/cyqcusm6tv5zt23b6vfsep5l4ijkywqce2oxtlni5dg5dvsqewaj.py
# Topologically Sorted Source Nodes: [pool_out_3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   pool_out_3 => add_10, clamp_max_4
# Graph fragment:
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_7, 1), kwargs = {})
#   %clamp_max_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_10, 15), kwargs = {})
triton_poi_fused_add_clamp_13 = async_compile.triton('triton_poi_fused_add_clamp_13', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_13(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 15, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5k/c5kja4apvr4z3obm5xr2bnpboliyzhzdx4tmujfdo7c4ejaqpqbr.py
# Topologically Sorted Source Nodes: [pool_out_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   pool_out_3 => add_9, clamp_max_6, clamp_min_4, clamp_min_6, convert_element_type_6, iota_2, mul_9, sub_10, sub_8
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 0.5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 1.0), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, 0.5), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_8, 0.0), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_4, %convert_element_type_9), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = triton_helpers.minimum(tmp12, tmp4)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/27/c27cdpres3mejhphaktmsy2ijrsb2fwykz6ptrryzrspcbrkfvp2.py
# Topologically Sorted Source Nodes: [pool_out_2, pool_out_3], Original ATen: [aten.max_pool2d_with_indices, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   pool_out_2 => _low_memory_max_pool2d_with_offsets_1
#   pool_out_3 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_13, add_14, mul_11, mul_12, sub_11, sub_12, sub_14
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%where, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem_2, [None, None, %convert_element_type_7, %convert_element_type_9]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem_2, [None, None, %convert_element_type_7, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem_2, [None, None, %clamp_max_4, %convert_element_type_9]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem_2, [None, None, %clamp_max_4, %clamp_max_5]), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %clamp_max_6), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_11), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %clamp_max_6), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_12), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_14, %add_13), kwargs = {})
triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_15 = async_compile.triton('triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = ((xindex // 256) % 16)
    x3 = xindex // 4096
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (x2 + 32*tmp8 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (16 + x2 + 32*tmp8 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.load(in_ptr2 + (512 + x2 + 32*tmp8 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.load(in_ptr2 + (528 + x2 + 32*tmp8 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp17 = tmp16 + tmp1
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr2 + (x2 + 32*tmp8 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (16 + x2 + 32*tmp8 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.load(in_ptr2 + (512 + x2 + 32*tmp8 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = tl.load(in_ptr2 + (528 + x2 + 32*tmp8 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp28 = tmp27 + tmp1
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr2 + (x2 + 32*tmp30 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (16 + x2 + 32*tmp30 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tl.load(in_ptr2 + (512 + x2 + 32*tmp30 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tmp36 = tl.load(in_ptr2 + (528 + x2 + 32*tmp30 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp37 = triton_helpers.maximum(tmp36, tmp35)
    tmp38 = tmp37 - tmp26
    tmp40 = tmp38 * tmp39
    tmp41 = tmp26 + tmp40
    tmp42 = tl.load(in_ptr2 + (x2 + 32*tmp30 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr2 + (16 + x2 + 32*tmp30 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tmp45 = tl.load(in_ptr2 + (512 + x2 + 32*tmp30 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.load(in_ptr2 + (528 + x2 + 32*tmp30 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tmp49 = tmp48 - tmp15
    tmp50 = tmp49 * tmp39
    tmp51 = tmp15 + tmp50
    tmp52 = tmp51 - tmp41
    tl.store(in_out_ptr0 + (x5), tmp41, None)
    tl.store(in_out_ptr1 + (x5), tmp52, None)
''', device_str='cuda')


# kernel path: inductor_cache/zd/czdfz2lynmwlabmbsweabcem4gx7eh4spzii3x6fbonpbbmzmdtm.py
# Topologically Sorted Source Nodes: [output_3, output_4, output_5], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_3 => cat_1
#   output_4 => add_17, mul_15, mul_16, sub_15
#   output_5 => gt_1, mul_17, where_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_1, %add_15], 1), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %unsqueeze_9), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_11), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %unsqueeze_13), kwargs = {})
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %unsqueeze_15), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_17, 0), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %add_17), kwargs = {})
#   %where_1 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_17, %mul_17), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x4 = xindex // 64
    x3 = xindex // 16384
    x6 = ((xindex // 64) % 256)
    x2 = ((xindex // 1024) % 16)
    x5 = xindex
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr8 + (0))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 48, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (48*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 64, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x6 + 256*((-48) + x0) + 4096*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x6 + 256*((-48) + x0) + 4096*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr3 + (x2), tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0
    tmp33 = tmp31 > tmp32
    tmp36 = tmp35 * tmp31
    tmp37 = tl.where(tmp33, tmp31, tmp36)
    tl.store(out_ptr0 + (x5), tmp16, None)
    tl.store(in_out_ptr0 + (x5), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/me/cmepiyprnc2kix47silqzbr42qgbvczi2ukiyneuda25jiyrm5ns.py
# Topologically Sorted Source Nodes: [output_6, output_7], Original ATen: [aten.convolution, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_6 => convolution_2
#   output_7 => gt_2, mul_18, where_2
# Graph fragment:
#   %convolution_2 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%where_1, %primals_14, %primals_15, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %convolution_2), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convolution_2, %mul_18), kwargs = {})
triton_poi_fused__prelu_kernel_convolution_17 = async_compile.triton('triton_poi_fused__prelu_kernel_convolution_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_convolution_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_convolution_17(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp7 = tmp6 * tmp2
    tmp8 = tl.where(tmp4, tmp2, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/2j/c2jnmluvbr5pp7gr3crkih7v6ftrbsb2lbanexnbyljslexjrvmy.py
# Topologically Sorted Source Nodes: [output_8, output_9, output_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_10 => gt_3, mul_22, where_3
#   output_8 => convolution_3
#   output_9 => add_19, mul_20, mul_21, sub_16
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_2, %primals_17, %primals_18, [1, 1], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_17), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_19), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_21), kwargs = {})
#   %add_19 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %unsqueeze_23), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_19, 0), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %add_19), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_19, %mul_22), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp20 = tl.load(in_ptr5 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp22 = tmp21 * tmp17
    tmp23 = tl.where(tmp19, tmp17, tmp22)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(in_out_ptr1 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/nb/cnbzo7akwiy6wnzwta3vjxvxgdmgumaiz3jbgg3ooop22oosff4z.py
# Topologically Sorted Source Nodes: [output_13, output_14, add, output_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
# Source node to ATen node mapping:
#   add => add_22
#   output_13 => convolution_5
#   output_14 => add_21, mul_25, mul_26, sub_17
#   output_16 => gt_5, mul_27, where_5
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_4, %primals_25, %primals_26, [1, 1], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_25), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_27), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_29), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_31), kwargs = {})
#   %add_22 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %where_1), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_22, 0), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %add_22), kwargs = {})
#   %where_5 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_22, %mul_27), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x2), None)
    tmp22 = tl.load(in_ptr6 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
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
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp19 > tmp20
    tmp24 = tmp23 * tmp19
    tmp25 = tl.where(tmp21, tmp19, tmp24)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/hb/chbah7rsfk3m3buvqzuhxfyr2wp67tb52v72mjzo3pmz3jcykejg.py
# Topologically Sorted Source Nodes: [pool_out_4], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   pool_out_4 => getitem_5
# Graph fragment:
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_20 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 8)
    x2 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 2048*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 2048*x2), None)
    tmp7 = tl.load(in_ptr0 + (1024 + x0 + 128*x1 + 2048*x2), None)
    tmp12 = tl.load(in_ptr0 + (1088 + x0 + 128*x1 + 2048*x2), None)
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


# kernel path: inductor_cache/q5/cq52qe4lm7lzbvt5e4xab62ogqmi54qbgpsenk23jvlg7cvjl7q5.py
# Topologically Sorted Source Nodes: [pool_out_5], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   pool_out_5 => convert_element_type_33
# Graph fragment:
#   %convert_element_type_33 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_26, torch.int64), kwargs = {})
triton_poi_fused__to_copy_21 = async_compile.triton('triton_poi_fused__to_copy_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cm/ccm4cwricj6n5zg4txtxiq53zq2nf5266cyxymv7iam4mmgouefy.py
# Topologically Sorted Source Nodes: [pool_out_5], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   pool_out_5 => add_44, clamp_max_8
# Graph fragment:
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_33, 1), kwargs = {})
#   %clamp_max_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_44, 7), kwargs = {})
triton_poi_fused_add_clamp_22 = async_compile.triton('triton_poi_fused_add_clamp_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_22(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 7, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pd/cpdq4fmlkrssrg23ut4tnxiqkvah3bybfrmdmt4s5qc7lheqcfcx.py
# Topologically Sorted Source Nodes: [pool_out_5], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   pool_out_5 => add_43, clamp_max_10, clamp_min_10, clamp_min_8, convert_element_type_32, iota_4, mul_68, sub_26, sub_28
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_4, torch.float32), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_32, 0.5), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, 1.0), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_68, 0.5), kwargs = {})
#   %clamp_min_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_26, 0.0), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_8, %convert_element_type_35), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_28, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = triton_helpers.minimum(tmp12, tmp4)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z7/cz7w6quofihvcbhmzpy3qfj7ravjyoeuw6gzvlmxizv3dojofvgt.py
# Topologically Sorted Source Nodes: [pool_out_4, pool_out_5], Original ATen: [aten.max_pool2d_with_indices, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   pool_out_4 => _low_memory_max_pool2d_with_offsets_2
#   pool_out_5 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_47, add_48, mul_70, mul_71, sub_29, sub_30, sub_32
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%where_21, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem_4, [None, None, %convert_element_type_33, %convert_element_type_35]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem_4, [None, None, %convert_element_type_33, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem_4, [None, None, %clamp_max_8, %convert_element_type_35]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem_4, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %clamp_max_10), kwargs = {})
#   %add_47 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_70), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %clamp_max_10), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_71), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_48, %add_47), kwargs = {})
triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_24 = async_compile.triton('triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_24(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = ((xindex // 64) % 64)
    x3 = xindex // 4096
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (x2 + 128*tmp8 + 2048*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (64 + x2 + 128*tmp8 + 2048*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.load(in_ptr2 + (1024 + x2 + 128*tmp8 + 2048*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.load(in_ptr2 + (1088 + x2 + 128*tmp8 + 2048*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp17 = tmp16 + tmp1
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr2 + (x2 + 128*tmp8 + 2048*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (64 + x2 + 128*tmp8 + 2048*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.load(in_ptr2 + (1024 + x2 + 128*tmp8 + 2048*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = tl.load(in_ptr2 + (1088 + x2 + 128*tmp8 + 2048*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp28 = tmp27 + tmp1
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr2 + (x2 + 128*tmp30 + 2048*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (64 + x2 + 128*tmp30 + 2048*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tl.load(in_ptr2 + (1024 + x2 + 128*tmp30 + 2048*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tmp36 = tl.load(in_ptr2 + (1088 + x2 + 128*tmp30 + 2048*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp37 = triton_helpers.maximum(tmp36, tmp35)
    tmp38 = tmp37 - tmp26
    tmp40 = tmp38 * tmp39
    tmp41 = tmp26 + tmp40
    tmp42 = tl.load(in_ptr2 + (x2 + 128*tmp30 + 2048*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr2 + (64 + x2 + 128*tmp30 + 2048*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tmp45 = tl.load(in_ptr2 + (1024 + x2 + 128*tmp30 + 2048*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.load(in_ptr2 + (1088 + x2 + 128*tmp30 + 2048*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tmp49 = tmp48 - tmp15
    tmp50 = tmp49 * tmp39
    tmp51 = tmp15 + tmp50
    tmp52 = tmp51 - tmp41
    tl.store(in_out_ptr0 + (x5), tmp41, None)
    tl.store(in_out_ptr1 + (x5), tmp52, None)
''', device_str='cuda')


# kernel path: inductor_cache/vi/cvixqli3an445gnsw72mnhmn3x56ucwtsw2jigmatiwehzqddkw2.py
# Topologically Sorted Source Nodes: [output_61, output_62, output_63], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_61 => cat_2
#   output_62 => add_51, mul_74, mul_75, sub_33
#   output_63 => gt_22, mul_76, where_22
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_22, %add_49], 1), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_2, %unsqueeze_97), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_99), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_74, %unsqueeze_101), kwargs = {})
#   %add_51 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %unsqueeze_103), kwargs = {})
#   %gt_22 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_51, 0), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, %add_51), kwargs = {})
#   %where_22 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_22, %add_51, %mul_76), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x4 = xindex // 128
    x3 = xindex // 8192
    x6 = ((xindex // 128) % 64)
    x2 = ((xindex // 1024) % 8)
    x5 = xindex
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr8 + (0))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x6 + 64*((-64) + x0) + 4096*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x6 + 64*((-64) + x0) + 4096*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr3 + (x2), tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0
    tmp33 = tmp31 > tmp32
    tmp36 = tmp35 * tmp31
    tmp37 = tl.where(tmp33, tmp31, tmp36)
    tl.store(out_ptr0 + (x5), tmp16, None)
    tl.store(in_out_ptr0 + (x5), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/kj/ckjwt2uer7huru7aj3bwcytqajakqqhiokg4wtwgmwhbx6w7yway.py
# Topologically Sorted Source Nodes: [output_64, output_65], Original ATen: [aten.convolution, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_64 => convolution_23
#   output_65 => gt_23, mul_77, where_23
# Graph fragment:
#   %convolution_23 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%where_22, %primals_105, %primals_106, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_23 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_23, 0), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_29, %convolution_23), kwargs = {})
#   %where_23 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_23, %convolution_23, %mul_77), kwargs = {})
triton_poi_fused__prelu_kernel_convolution_26 = async_compile.triton('triton_poi_fused__prelu_kernel_convolution_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_convolution_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_convolution_26(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp7 = tmp6 * tmp2
    tmp8 = tl.where(tmp4, tmp2, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/2z/c2zzjugz3ej27f76ss5klkt3ihoowfnt5n5iuznpfymwonlkrck3.py
# Topologically Sorted Source Nodes: [output_66, output_67, output_68], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_66 => convolution_24
#   output_67 => add_53, mul_79, mul_80, sub_34
#   output_68 => gt_24, mul_81, where_24
# Graph fragment:
#   %convolution_24 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_23, %primals_108, %primals_109, [1, 1], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_105), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_107), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_109), kwargs = {})
#   %add_53 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_111), kwargs = {})
#   %gt_24 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_53, 0), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_29, %add_53), kwargs = {})
#   %where_24 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_24, %add_53, %mul_81), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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
    tmp20 = tl.load(in_ptr5 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp22 = tmp21 * tmp17
    tmp23 = tl.where(tmp19, tmp17, tmp22)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(in_out_ptr1 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/g2/cg2i5whcarapgxr733rh7ziaqbjeyrm2qbv734jbihq7dactjcxi.py
# Topologically Sorted Source Nodes: [output_71, output_72, add_5, output_74], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
# Source node to ATen node mapping:
#   add_5 => add_56
#   output_71 => convolution_26
#   output_72 => add_55, mul_84, mul_85, sub_35
#   output_74 => gt_26, mul_86, where_26
# Graph fragment:
#   %convolution_26 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_25, %primals_116, %primals_117, [1, 1], [0, 2], [1, 2], False, [0, 0], 1), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_113), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_115), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %unsqueeze_117), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %unsqueeze_119), kwargs = {})
#   %add_56 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_55, %where_22), kwargs = {})
#   %gt_26 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_56, 0), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_29, %add_56), kwargs = {})
#   %where_26 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_26, %add_56, %mul_86), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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
    tmp18 = tl.load(in_ptr5 + (x2), None)
    tmp22 = tl.load(in_ptr6 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
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
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp19 > tmp20
    tmp24 = tmp23 * tmp19
    tmp25 = tl.where(tmp21, tmp19, tmp24)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/wa/cwa3ooxsztcx2eqfhh3hfbcyvxb4lfuec742ufrtsdc4dd2dwcv6.py
# Topologically Sorted Source Nodes: [output_177, output_178, output_179], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_177 => convolution_64
#   output_178 => add_105, mul_182, mul_183, sub_55
#   output_179 => gt_64, mul_184, where_64
# Graph fragment:
#   %convolution_64 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_63, %primals_282, %primals_283, [2, 2], [1, 1], [1, 1], True, [1, 1], 1), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_273), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_275), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_182, %unsqueeze_277), kwargs = {})
#   %add_105 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_183, %unsqueeze_279), kwargs = {})
#   %gt_64 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_105, 0), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_70, %add_105), kwargs = {})
#   %where_64 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_64, %add_105, %mul_184), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_29(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    tmp20 = tl.load(in_ptr5 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp22 = tmp21 * tmp17
    tmp23 = tl.where(tmp19, tmp17, tmp22)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(in_out_ptr1 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/ff/cffzo4duj5z4mxe52n4scpmpwmssne5fw7x3otzzowwsoytyi462.py
# Topologically Sorted Source Nodes: [output_180, output_181], Original ATen: [aten.convolution, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_180 => convolution_65
#   output_181 => gt_65, mul_185, where_65
# Graph fragment:
#   %convolution_65 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%where_64, %primals_289, %primals_290, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_65 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_65, 0), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_71, %convolution_65), kwargs = {})
#   %where_65 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_65, %convolution_65, %mul_185), kwargs = {})
triton_poi_fused__prelu_kernel_convolution_30 = async_compile.triton('triton_poi_fused__prelu_kernel_convolution_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_convolution_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_convolution_30(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp7 = tmp6 * tmp2
    tmp8 = tl.where(tmp4, tmp2, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/we/cwenbachk7fsh6e4mf3hve2tztf32uy362gd5flt6dlabcwtgftr.py
# Topologically Sorted Source Nodes: [output_187, output_188, add_15, output_190], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
# Source node to ATen node mapping:
#   add_15 => add_110
#   output_187 => convolution_68
#   output_188 => add_109, mul_192, mul_193, sub_57
#   output_190 => gt_68, mul_194, where_68
# Graph fragment:
#   %convolution_68 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_67, %primals_300, %primals_301, [1, 1], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_289), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_291), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_192, %unsqueeze_293), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_193, %unsqueeze_295), kwargs = {})
#   %add_110 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_109, %where_64), kwargs = {})
#   %gt_68 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_110, 0), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_71, %add_110), kwargs = {})
#   %where_68 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_68, %add_110, %mul_194), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    tmp18 = tl.load(in_ptr5 + (x2), None)
    tmp22 = tl.load(in_ptr6 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
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
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp19 > tmp20
    tmp24 = tmp23 * tmp19
    tmp25 = tl.where(tmp21, tmp19, tmp24)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/fq/cfqt4pmzsz6crjntf63rptlcqidr5wxw62teq4xaps3knv2ujq7j.py
# Topologically Sorted Source Nodes: [output_198, output_199, add_16, output_201], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
# Source node to ATen node mapping:
#   add_16 => add_115
#   output_198 => convolution_72
#   output_199 => add_114, mul_202, mul_203, sub_59
#   output_201 => gt_72, mul_204, where_72
# Graph fragment:
#   %convolution_72 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_71, %primals_317, %primals_318, [1, 1], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_305), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_307), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_309), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_311), kwargs = {})
#   %add_115 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_114, %where_68), kwargs = {})
#   %gt_72 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_115, 0), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_75, %add_115), kwargs = {})
#   %where_72 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_72, %add_115, %mul_204), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = (yindex % 1024)
    y3 = yindex // 1024
    tmp0 = tl.load(in_out_ptr0 + (x1 + 16*y0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1 + 16*y0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, YBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp19 > tmp20
    tmp24 = tmp23 * tmp19
    tmp25 = tl.where(tmp21, tmp19, tmp24)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 16*y0), tmp2, xmask)
    tl.store(out_ptr0 + (x1 + 16*y0), tmp19, xmask)
    tl.store(out_ptr1 + (y2 + 1024*x1 + 16384*y3), tmp25, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322 = args
    args.clear()
    assert_size_stride(primals_1, (13, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (1, ), (1, ))
    assert_size_stride(primals_8, (48, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (1, ), (1, ))
    assert_size_stride(primals_14, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (1, ), (1, ))
    assert_size_stride(primals_17, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_33, (1, ), (1, ))
    assert_size_stride(primals_34, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_48, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (1, ), (1, ))
    assert_size_stride(primals_51, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_52, (64, ), (1, ))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_58, (64, ), (1, ))
    assert_size_stride(primals_59, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (64, ), (1, ))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (1, ), (1, ))
    assert_size_stride(primals_68, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (1, ), (1, ))
    assert_size_stride(primals_85, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_86, (64, ), (1, ))
    assert_size_stride(primals_87, (64, ), (1, ))
    assert_size_stride(primals_88, (64, ), (1, ))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_92, (64, ), (1, ))
    assert_size_stride(primals_93, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_94, (64, ), (1, ))
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_96, (64, ), (1, ))
    assert_size_stride(primals_97, (64, ), (1, ))
    assert_size_stride(primals_98, (64, ), (1, ))
    assert_size_stride(primals_99, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (128, ), (1, ))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (1, ), (1, ))
    assert_size_stride(primals_105, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (1, ), (1, ))
    assert_size_stride(primals_108, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_117, (128, ), (1, ))
    assert_size_stride(primals_118, (128, ), (1, ))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (1, ), (1, ))
    assert_size_stride(primals_125, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_141, (1, ), (1, ))
    assert_size_stride(primals_142, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (128, ), (1, ))
    assert_size_stride(primals_148, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_149, (128, ), (1, ))
    assert_size_stride(primals_150, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (1, ), (1, ))
    assert_size_stride(primals_159, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_167, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (128, ), (1, ))
    assert_size_stride(primals_172, (128, ), (1, ))
    assert_size_stride(primals_173, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (1, ), (1, ))
    assert_size_stride(primals_176, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (128, ), (1, ))
    assert_size_stride(primals_190, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_191, (128, ), (1, ))
    assert_size_stride(primals_192, (1, ), (1, ))
    assert_size_stride(primals_193, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_194, (128, ), (1, ))
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (128, ), (1, ))
    assert_size_stride(primals_198, (128, ), (1, ))
    assert_size_stride(primals_199, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_200, (128, ), (1, ))
    assert_size_stride(primals_201, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_202, (128, ), (1, ))
    assert_size_stride(primals_203, (128, ), (1, ))
    assert_size_stride(primals_204, (128, ), (1, ))
    assert_size_stride(primals_205, (128, ), (1, ))
    assert_size_stride(primals_206, (128, ), (1, ))
    assert_size_stride(primals_207, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_208, (128, ), (1, ))
    assert_size_stride(primals_209, (1, ), (1, ))
    assert_size_stride(primals_210, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (128, ), (1, ))
    assert_size_stride(primals_214, (128, ), (1, ))
    assert_size_stride(primals_215, (128, ), (1, ))
    assert_size_stride(primals_216, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_219, (128, ), (1, ))
    assert_size_stride(primals_220, (128, ), (1, ))
    assert_size_stride(primals_221, (128, ), (1, ))
    assert_size_stride(primals_222, (128, ), (1, ))
    assert_size_stride(primals_223, (128, ), (1, ))
    assert_size_stride(primals_224, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (1, ), (1, ))
    assert_size_stride(primals_227, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_228, (128, ), (1, ))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, ), (1, ))
    assert_size_stride(primals_231, (128, ), (1, ))
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (128, 128, 3, 1), (384, 3, 1, 1))
    assert_size_stride(primals_234, (128, ), (1, ))
    assert_size_stride(primals_235, (128, 128, 1, 3), (384, 3, 3, 1))
    assert_size_stride(primals_236, (128, ), (1, ))
    assert_size_stride(primals_237, (128, ), (1, ))
    assert_size_stride(primals_238, (128, ), (1, ))
    assert_size_stride(primals_239, (128, ), (1, ))
    assert_size_stride(primals_240, (128, ), (1, ))
    assert_size_stride(primals_241, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_242, (64, ), (1, ))
    assert_size_stride(primals_243, (64, ), (1, ))
    assert_size_stride(primals_244, (64, ), (1, ))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (64, ), (1, ))
    assert_size_stride(primals_247, (1, ), (1, ))
    assert_size_stride(primals_248, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_249, (64, ), (1, ))
    assert_size_stride(primals_250, (1, ), (1, ))
    assert_size_stride(primals_251, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_252, (64, ), (1, ))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (64, ), (1, ))
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_260, (64, ), (1, ))
    assert_size_stride(primals_261, (64, ), (1, ))
    assert_size_stride(primals_262, (64, ), (1, ))
    assert_size_stride(primals_263, (64, ), (1, ))
    assert_size_stride(primals_264, (64, ), (1, ))
    assert_size_stride(primals_265, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_266, (64, ), (1, ))
    assert_size_stride(primals_267, (1, ), (1, ))
    assert_size_stride(primals_268, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_269, (64, ), (1, ))
    assert_size_stride(primals_270, (64, ), (1, ))
    assert_size_stride(primals_271, (64, ), (1, ))
    assert_size_stride(primals_272, (64, ), (1, ))
    assert_size_stride(primals_273, (64, ), (1, ))
    assert_size_stride(primals_274, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_275, (64, ), (1, ))
    assert_size_stride(primals_276, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_277, (64, ), (1, ))
    assert_size_stride(primals_278, (64, ), (1, ))
    assert_size_stride(primals_279, (64, ), (1, ))
    assert_size_stride(primals_280, (64, ), (1, ))
    assert_size_stride(primals_281, (64, ), (1, ))
    assert_size_stride(primals_282, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_283, (16, ), (1, ))
    assert_size_stride(primals_284, (16, ), (1, ))
    assert_size_stride(primals_285, (16, ), (1, ))
    assert_size_stride(primals_286, (16, ), (1, ))
    assert_size_stride(primals_287, (16, ), (1, ))
    assert_size_stride(primals_288, (1, ), (1, ))
    assert_size_stride(primals_289, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_290, (16, ), (1, ))
    assert_size_stride(primals_291, (1, ), (1, ))
    assert_size_stride(primals_292, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_293, (16, ), (1, ))
    assert_size_stride(primals_294, (16, ), (1, ))
    assert_size_stride(primals_295, (16, ), (1, ))
    assert_size_stride(primals_296, (16, ), (1, ))
    assert_size_stride(primals_297, (16, ), (1, ))
    assert_size_stride(primals_298, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_299, (16, ), (1, ))
    assert_size_stride(primals_300, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_301, (16, ), (1, ))
    assert_size_stride(primals_302, (16, ), (1, ))
    assert_size_stride(primals_303, (16, ), (1, ))
    assert_size_stride(primals_304, (16, ), (1, ))
    assert_size_stride(primals_305, (16, ), (1, ))
    assert_size_stride(primals_306, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_307, (16, ), (1, ))
    assert_size_stride(primals_308, (1, ), (1, ))
    assert_size_stride(primals_309, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_310, (16, ), (1, ))
    assert_size_stride(primals_311, (16, ), (1, ))
    assert_size_stride(primals_312, (16, ), (1, ))
    assert_size_stride(primals_313, (16, ), (1, ))
    assert_size_stride(primals_314, (16, ), (1, ))
    assert_size_stride(primals_315, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_316, (16, ), (1, ))
    assert_size_stride(primals_317, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_318, (16, ), (1, ))
    assert_size_stride(primals_319, (16, ), (1, ))
    assert_size_stride(primals_320, (16, ), (1, ))
    assert_size_stride(primals_321, (16, ), (1, ))
    assert_size_stride(primals_322, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((13, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 39, 9, grid=grid(39, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((48, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_8, buf2, 768, 9, grid=grid(768, 9), stream=stream0)
        del primals_8
        buf3 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_14, buf3, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_14
        buf4 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_17, buf4, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_17
        buf5 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_23, buf5, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_23
        buf6 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_25, buf6, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_25
        buf7 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_31, buf7, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_31
        buf8 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_34, buf8, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_34
        buf9 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_40, buf9, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_40
        buf10 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_42, buf10, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_42
        buf11 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_48, buf11, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_48
        buf12 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_51, buf12, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_51
        buf13 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_57, buf13, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_57
        buf14 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_59, buf14, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_59
        buf15 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_65, buf15, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_65
        buf16 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_68, buf16, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_68
        buf17 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_74, buf17, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_74
        buf18 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_76, buf18, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_76
        buf19 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_82, buf19, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_82
        buf20 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_85, buf20, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_85
        buf21 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_91, buf21, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_91
        buf22 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_93, buf22, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_93
        buf23 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_99, buf23, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_99
        buf24 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_105, buf24, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_105
        buf25 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_108, buf25, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_108
        buf26 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_114, buf26, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_114
        buf27 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_116, buf27, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_116
        buf28 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_122, buf28, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_122
        buf29 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_125, buf29, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_125
        buf30 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_131, buf30, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_131
        buf31 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_133, buf31, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_133
        buf32 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_139, buf32, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_139
        buf33 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_142, buf33, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_142
        buf34 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_148, buf34, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_148
        buf35 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_150, buf35, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_150
        buf36 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_156, buf36, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_156
        buf37 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_159, buf37, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_159
        buf38 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_165, buf38, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_165
        buf39 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_167, buf39, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_167
        buf40 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_173, buf40, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_173
        buf41 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_176, buf41, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_176
        buf42 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_182, buf42, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_182
        buf43 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_184, buf43, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_184
        buf44 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_190, buf44, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_190
        buf45 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_193, buf45, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_193
        buf46 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_199, buf46, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_199
        buf47 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_201, buf47, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_201
        buf48 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_207, buf48, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_207
        buf49 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_210, buf49, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_210
        buf50 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_216, buf50, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_216
        buf51 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_218, buf51, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_218
        buf52 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_224, buf52, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_224
        buf53 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_227, buf53, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_227
        buf54 = empty_strided_cuda((128, 128, 3, 1), (384, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_233, buf54, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_233
        buf55 = empty_strided_cuda((128, 128, 1, 3), (384, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_235, buf55, 16384, 3, grid=grid(16384, 3), stream=stream0)
        del primals_235
        buf56 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_241, buf56, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_241
        buf57 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_248, buf57, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_248
        buf58 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_251, buf58, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_251
        buf59 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_257, buf59, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_257
        buf60 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_259, buf60, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_259
        buf61 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_265, buf61, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_265
        buf62 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_268, buf62, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_268
        buf63 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_274, buf63, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_274
        buf64 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_276, buf64, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_276
        buf65 = empty_strided_cuda((64, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_282, buf65, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_282
        buf66 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_289, buf66, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_289
        buf67 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_292, buf67, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_292
        buf68 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_298, buf68, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_298
        buf69 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_300, buf69, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_300
        buf70 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_306, buf70, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_306
        buf71 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_309, buf71, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_309
        buf72 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_315, buf72, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_315
        buf73 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_317, buf73, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_317
        # Topologically Sorted Source Nodes: [conv_out], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 13, 32, 32), (13312, 1, 416, 13))
        buf75 = empty_strided_cuda((4, 3, 32, 32), (3072, 1024, 32, 1), torch.float32)
        buf77 = empty_strided_cuda((4, 3, 32, 32), (3072, 1024, 32, 1), torch.float32)
        buf76 = buf75; del buf75  # reuse
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [pool_out, pool_out_1], Original ATen: [aten.max_pool2d_with_indices, aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_max_pool2d_with_indices_mul_sub_9.run(buf76, buf78, buf1, 12288, grid=grid(12288), stream=stream0)
        buf79 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf80 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [output, output_1, output_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_10.run(buf81, buf74, buf76, buf78, primals_3, primals_4, primals_5, primals_6, primals_7, buf79, 65536, grid=grid(65536), stream=stream0)
        del buf74
        del buf76
        del buf78
        # Topologically Sorted Source Nodes: [conv_out_1], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf83 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.int8)
        # Topologically Sorted Source Nodes: [pool_out_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_11.run(buf81, buf83, 16384, grid=grid(16384), stream=stream0)
        buf84 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pool_out_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf84, 16, grid=grid(16), stream=stream0)
        buf85 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pool_out_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_13.run(buf85, 16, grid=grid(16), stream=stream0)
        buf86 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pool_out_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf86, 16, grid=grid(16), stream=stream0)
        buf87 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pool_out_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_13.run(buf87, 16, grid=grid(16), stream=stream0)
        buf90 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [pool_out_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14.run(buf90, 16, grid=grid(16), stream=stream0)
        buf89 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf88 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf91 = buf88; del buf88  # reuse
        buf93 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [pool_out_2, pool_out_3], Original ATen: [aten.max_pool2d_with_indices, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_15.run(buf91, buf93, buf85, buf86, buf81, buf84, buf87, buf90, 16384, grid=grid(16384), stream=stream0)
        buf92 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pool_out_3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14.run(buf92, 16, grid=grid(16), stream=stream0)
        buf94 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf95 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [output_3, output_4, output_5], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_16.run(buf96, buf82, buf91, buf93, buf92, primals_9, primals_10, primals_11, primals_12, primals_13, buf94, 65536, grid=grid(65536), stream=stream0)
        del buf82
        # Topologically Sorted Source Nodes: [output_6], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, buf3, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf98 = buf97; del buf97  # reuse
        buf99 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_6, output_7], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf98, primals_15, primals_16, buf99, 65536, grid=grid(65536), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [output_8], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, buf4, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf101 = buf100; del buf100  # reuse
        buf102 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [output_8, output_9, output_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18.run(buf101, buf103, primals_18, primals_19, primals_20, primals_21, primals_22, primals_16, 65536, grid=grid(65536), stream=stream0)
        del primals_18
        # Topologically Sorted Source Nodes: [output_11], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, buf5, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf105 = buf104; del buf104  # reuse
        buf106 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_11, output_12], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf105, primals_24, primals_16, buf106, 65536, grid=grid(65536), stream=stream0)
        del primals_24
        # Topologically Sorted Source Nodes: [output_13], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, buf6, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf108 = buf107; del buf107  # reuse
        buf109 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf110 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_13, output_14, add, output_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19.run(buf108, primals_26, primals_27, primals_28, primals_29, primals_30, buf96, primals_16, buf109, buf110, 65536, grid=grid(65536), stream=stream0)
        del primals_26
        del primals_30
        # Topologically Sorted Source Nodes: [output_17], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, buf7, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf112 = buf111; del buf111  # reuse
        buf113 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_17, output_18], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf112, primals_32, primals_33, buf113, 65536, grid=grid(65536), stream=stream0)
        del primals_32
        # Topologically Sorted Source Nodes: [output_19], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, buf8, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf115 = buf114; del buf114  # reuse
        buf116 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [output_19, output_20, output_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18.run(buf115, buf117, primals_35, primals_36, primals_37, primals_38, primals_39, primals_33, 65536, grid=grid(65536), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [output_22], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, buf9, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf119 = buf118; del buf118  # reuse
        buf120 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_22, output_23], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf119, primals_41, primals_33, buf120, 65536, grid=grid(65536), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [output_24], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, buf10, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf122 = buf121; del buf121  # reuse
        buf123 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf124 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_24, output_25, add_1, output_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19.run(buf122, primals_43, primals_44, primals_45, primals_46, primals_47, buf110, primals_33, buf123, buf124, 65536, grid=grid(65536), stream=stream0)
        del primals_43
        del primals_47
        # Topologically Sorted Source Nodes: [output_28], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, buf11, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf126 = buf125; del buf125  # reuse
        buf127 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_28, output_29], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf126, primals_49, primals_50, buf127, 65536, grid=grid(65536), stream=stream0)
        del primals_49
        # Topologically Sorted Source Nodes: [output_30], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, buf12, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf129 = buf128; del buf128  # reuse
        buf130 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [output_30, output_31, output_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18.run(buf129, buf131, primals_52, primals_53, primals_54, primals_55, primals_56, primals_50, 65536, grid=grid(65536), stream=stream0)
        del primals_52
        # Topologically Sorted Source Nodes: [output_33], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, buf13, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf133 = buf132; del buf132  # reuse
        buf134 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_33, output_34], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf133, primals_58, primals_50, buf134, 65536, grid=grid(65536), stream=stream0)
        del primals_58
        # Topologically Sorted Source Nodes: [output_35], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, buf14, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf136 = buf135; del buf135  # reuse
        buf137 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf138 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_35, output_36, add_2, output_38], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19.run(buf136, primals_60, primals_61, primals_62, primals_63, primals_64, buf124, primals_50, buf137, buf138, 65536, grid=grid(65536), stream=stream0)
        del primals_60
        del primals_64
        # Topologically Sorted Source Nodes: [output_39], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, buf15, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf140 = buf139; del buf139  # reuse
        buf141 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_39, output_40], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf140, primals_66, primals_67, buf141, 65536, grid=grid(65536), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [output_41], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, buf16, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf143 = buf142; del buf142  # reuse
        buf144 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf145 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [output_41, output_42, output_43], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18.run(buf143, buf145, primals_69, primals_70, primals_71, primals_72, primals_73, primals_67, 65536, grid=grid(65536), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [output_44], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, buf17, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf147 = buf146; del buf146  # reuse
        buf148 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_44, output_45], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf147, primals_75, primals_67, buf148, 65536, grid=grid(65536), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [output_46], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, buf18, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf150 = buf149; del buf149  # reuse
        buf151 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf152 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_46, output_47, add_3, output_49], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19.run(buf150, primals_77, primals_78, primals_79, primals_80, primals_81, buf138, primals_67, buf151, buf152, 65536, grid=grid(65536), stream=stream0)
        del primals_77
        del primals_81
        # Topologically Sorted Source Nodes: [output_50], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, buf19, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf154 = buf153; del buf153  # reuse
        buf155 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_50, output_51], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf154, primals_83, primals_84, buf155, 65536, grid=grid(65536), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [output_52], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, buf20, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf157 = buf156; del buf156  # reuse
        buf158 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [output_52, output_53, output_54], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18.run(buf157, buf159, primals_86, primals_87, primals_88, primals_89, primals_90, primals_84, 65536, grid=grid(65536), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [output_55], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, buf21, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf161 = buf160; del buf160  # reuse
        buf162 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_55, output_56], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf161, primals_92, primals_84, buf162, 65536, grid=grid(65536), stream=stream0)
        del primals_92
        # Topologically Sorted Source Nodes: [output_57], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, buf22, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf164 = buf163; del buf163  # reuse
        buf165 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf166 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_57, output_58, add_4, output_60], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19.run(buf164, primals_94, primals_95, primals_96, primals_97, primals_98, buf152, primals_84, buf165, buf166, 65536, grid=grid(65536), stream=stream0)
        del primals_94
        del primals_98
        # Topologically Sorted Source Nodes: [conv_out_2], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, buf23, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf168 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.int8)
        # Topologically Sorted Source Nodes: [pool_out_4], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_20.run(buf166, buf168, 16384, grid=grid(16384), stream=stream0)
        buf169 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pool_out_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(buf169, 8, grid=grid(8), stream=stream0)
        buf170 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pool_out_5], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_22.run(buf170, 8, grid=grid(8), stream=stream0)
        buf171 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pool_out_5], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(buf171, 8, grid=grid(8), stream=stream0)
        buf172 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pool_out_5], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_22.run(buf172, 8, grid=grid(8), stream=stream0)
        buf175 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [pool_out_5], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23.run(buf175, 8, grid=grid(8), stream=stream0)
        buf174 = reinterpret_tensor(buf93, (4, 64, 8, 8), (4096, 64, 8, 1), 0); del buf93  # reuse
        buf173 = reinterpret_tensor(buf91, (4, 64, 8, 8), (4096, 64, 8, 1), 0); del buf91  # reuse
        buf176 = buf173; del buf173  # reuse
        buf178 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [pool_out_4, pool_out_5], Original ATen: [aten.max_pool2d_with_indices, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_24.run(buf176, buf178, buf170, buf171, buf166, buf169, buf172, buf175, 16384, grid=grid(16384), stream=stream0)
        buf177 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pool_out_5], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23.run(buf177, 8, grid=grid(8), stream=stream0)
        buf179 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf180 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf181 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [output_61, output_62, output_63], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25.run(buf181, buf167, buf176, buf178, buf177, primals_100, primals_101, primals_102, primals_103, primals_104, buf179, 32768, grid=grid(32768), stream=stream0)
        del buf167
        del buf176
        del buf178
        # Topologically Sorted Source Nodes: [output_64], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, buf24, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf183 = buf182; del buf182  # reuse
        buf184 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_64, output_65], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf183, primals_106, primals_107, buf184, 32768, grid=grid(32768), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [output_66], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, buf25, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf186 = buf185; del buf185  # reuse
        buf187 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf188 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [output_66, output_67, output_68], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27.run(buf186, buf188, primals_109, primals_110, primals_111, primals_112, primals_113, primals_107, 32768, grid=grid(32768), stream=stream0)
        del primals_109
        # Topologically Sorted Source Nodes: [output_69], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, buf26, stride=(1, 1), padding=(2, 0), dilation=(2, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf190 = buf189; del buf189  # reuse
        buf191 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_69, output_70], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf190, primals_115, primals_107, buf191, 32768, grid=grid(32768), stream=stream0)
        del primals_115
        # Topologically Sorted Source Nodes: [output_71], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, buf27, stride=(1, 1), padding=(0, 2), dilation=(1, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf193 = buf192; del buf192  # reuse
        buf194 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf195 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_71, output_72, add_5, output_74], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28.run(buf193, primals_117, primals_118, primals_119, primals_120, primals_121, buf181, primals_107, buf194, buf195, 32768, grid=grid(32768), stream=stream0)
        del primals_117
        del primals_121
        # Topologically Sorted Source Nodes: [output_75], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, buf28, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf197 = buf196; del buf196  # reuse
        buf198 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_75, output_76], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf197, primals_123, primals_124, buf198, 32768, grid=grid(32768), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [output_77], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, buf29, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf200 = buf199; del buf199  # reuse
        buf201 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf202 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [output_77, output_78, output_79], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27.run(buf200, buf202, primals_126, primals_127, primals_128, primals_129, primals_130, primals_124, 32768, grid=grid(32768), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [output_80], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, buf30, stride=(1, 1), padding=(4, 0), dilation=(4, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf204 = buf203; del buf203  # reuse
        buf205 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_80, output_81], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf204, primals_132, primals_124, buf205, 32768, grid=grid(32768), stream=stream0)
        del primals_132
        # Topologically Sorted Source Nodes: [output_82], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, buf31, stride=(1, 1), padding=(0, 4), dilation=(1, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf207 = buf206; del buf206  # reuse
        buf208 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf209 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_82, output_83, add_6, output_85], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28.run(buf207, primals_134, primals_135, primals_136, primals_137, primals_138, buf195, primals_124, buf208, buf209, 32768, grid=grid(32768), stream=stream0)
        del primals_134
        del primals_138
        # Topologically Sorted Source Nodes: [output_86], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, buf32, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf211 = buf210; del buf210  # reuse
        buf212 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_86, output_87], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf211, primals_140, primals_141, buf212, 32768, grid=grid(32768), stream=stream0)
        del primals_140
        # Topologically Sorted Source Nodes: [output_88], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, buf33, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf214 = buf213; del buf213  # reuse
        buf215 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf216 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [output_88, output_89, output_90], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27.run(buf214, buf216, primals_143, primals_144, primals_145, primals_146, primals_147, primals_141, 32768, grid=grid(32768), stream=stream0)
        del primals_143
        # Topologically Sorted Source Nodes: [output_91], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, buf34, stride=(1, 1), padding=(8, 0), dilation=(8, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf218 = buf217; del buf217  # reuse
        buf219 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_91, output_92], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf218, primals_149, primals_141, buf219, 32768, grid=grid(32768), stream=stream0)
        del primals_149
        # Topologically Sorted Source Nodes: [output_93], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, buf35, stride=(1, 1), padding=(0, 8), dilation=(1, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf221 = buf220; del buf220  # reuse
        buf222 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf223 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_93, output_94, add_7, output_96], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28.run(buf221, primals_151, primals_152, primals_153, primals_154, primals_155, buf209, primals_141, buf222, buf223, 32768, grid=grid(32768), stream=stream0)
        del primals_151
        del primals_155
        # Topologically Sorted Source Nodes: [output_97], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, buf36, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf225 = buf224; del buf224  # reuse
        buf226 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_97, output_98], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf225, primals_157, primals_158, buf226, 32768, grid=grid(32768), stream=stream0)
        del primals_157
        # Topologically Sorted Source Nodes: [output_99], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, buf37, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf228 = buf227; del buf227  # reuse
        buf229 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf230 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [output_99, output_100, output_101], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27.run(buf228, buf230, primals_160, primals_161, primals_162, primals_163, primals_164, primals_158, 32768, grid=grid(32768), stream=stream0)
        del primals_160
        # Topologically Sorted Source Nodes: [output_102], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, buf38, stride=(1, 1), padding=(16, 0), dilation=(16, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf232 = buf231; del buf231  # reuse
        buf233 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_102, output_103], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf232, primals_166, primals_158, buf233, 32768, grid=grid(32768), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [output_104], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, buf39, stride=(1, 1), padding=(0, 16), dilation=(1, 16), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf235 = buf234; del buf234  # reuse
        buf236 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf237 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_104, output_105, add_8, output_107], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28.run(buf235, primals_168, primals_169, primals_170, primals_171, primals_172, buf223, primals_158, buf236, buf237, 32768, grid=grid(32768), stream=stream0)
        del primals_168
        del primals_172
        # Topologically Sorted Source Nodes: [output_108], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, buf40, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf239 = buf238; del buf238  # reuse
        buf240 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_108, output_109], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf239, primals_174, primals_175, buf240, 32768, grid=grid(32768), stream=stream0)
        del primals_174
        # Topologically Sorted Source Nodes: [output_110], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, buf41, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf242 = buf241; del buf241  # reuse
        buf243 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf244 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [output_110, output_111, output_112], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27.run(buf242, buf244, primals_177, primals_178, primals_179, primals_180, primals_181, primals_175, 32768, grid=grid(32768), stream=stream0)
        del primals_177
        # Topologically Sorted Source Nodes: [output_113], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, buf42, stride=(1, 1), padding=(2, 0), dilation=(2, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf246 = buf245; del buf245  # reuse
        buf247 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_113, output_114], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf246, primals_183, primals_175, buf247, 32768, grid=grid(32768), stream=stream0)
        del primals_183
        # Topologically Sorted Source Nodes: [output_115], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, buf43, stride=(1, 1), padding=(0, 2), dilation=(1, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf249 = buf248; del buf248  # reuse
        buf250 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf251 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_115, output_116, add_9, output_118], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28.run(buf249, primals_185, primals_186, primals_187, primals_188, primals_189, buf237, primals_175, buf250, buf251, 32768, grid=grid(32768), stream=stream0)
        del primals_185
        del primals_189
        # Topologically Sorted Source Nodes: [output_119], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, buf44, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf253 = buf252; del buf252  # reuse
        buf254 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_119, output_120], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf253, primals_191, primals_192, buf254, 32768, grid=grid(32768), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [output_121], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, buf45, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf256 = buf255; del buf255  # reuse
        buf257 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [output_121, output_122, output_123], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27.run(buf256, buf258, primals_194, primals_195, primals_196, primals_197, primals_198, primals_192, 32768, grid=grid(32768), stream=stream0)
        del primals_194
        # Topologically Sorted Source Nodes: [output_124], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, buf46, stride=(1, 1), padding=(4, 0), dilation=(4, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf260 = buf259; del buf259  # reuse
        buf261 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_124, output_125], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf260, primals_200, primals_192, buf261, 32768, grid=grid(32768), stream=stream0)
        del primals_200
        # Topologically Sorted Source Nodes: [output_126], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, buf47, stride=(1, 1), padding=(0, 4), dilation=(1, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf263 = buf262; del buf262  # reuse
        buf264 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf265 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_126, output_127, add_10, output_129], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28.run(buf263, primals_202, primals_203, primals_204, primals_205, primals_206, buf251, primals_192, buf264, buf265, 32768, grid=grid(32768), stream=stream0)
        del primals_202
        del primals_206
        # Topologically Sorted Source Nodes: [output_130], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, buf48, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf267 = buf266; del buf266  # reuse
        buf268 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_130, output_131], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf267, primals_208, primals_209, buf268, 32768, grid=grid(32768), stream=stream0)
        del primals_208
        # Topologically Sorted Source Nodes: [output_132], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, buf49, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf270 = buf269; del buf269  # reuse
        buf271 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf272 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [output_132, output_133, output_134], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27.run(buf270, buf272, primals_211, primals_212, primals_213, primals_214, primals_215, primals_209, 32768, grid=grid(32768), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [output_135], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, buf50, stride=(1, 1), padding=(8, 0), dilation=(8, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf274 = buf273; del buf273  # reuse
        buf275 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_135, output_136], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf274, primals_217, primals_209, buf275, 32768, grid=grid(32768), stream=stream0)
        del primals_217
        # Topologically Sorted Source Nodes: [output_137], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, buf51, stride=(1, 1), padding=(0, 8), dilation=(1, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf277 = buf276; del buf276  # reuse
        buf278 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf279 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_137, output_138, add_11, output_140], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28.run(buf277, primals_219, primals_220, primals_221, primals_222, primals_223, buf265, primals_209, buf278, buf279, 32768, grid=grid(32768), stream=stream0)
        del primals_219
        del primals_223
        # Topologically Sorted Source Nodes: [output_141], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, buf52, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf281 = buf280; del buf280  # reuse
        buf282 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_141, output_142], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf281, primals_225, primals_226, buf282, 32768, grid=grid(32768), stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [output_143], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, buf53, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf284 = buf283; del buf283  # reuse
        buf285 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf286 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [output_143, output_144, output_145], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_27.run(buf284, buf286, primals_228, primals_229, primals_230, primals_231, primals_232, primals_226, 32768, grid=grid(32768), stream=stream0)
        del primals_228
        # Topologically Sorted Source Nodes: [output_146], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, buf54, stride=(1, 1), padding=(16, 0), dilation=(16, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf288 = buf287; del buf287  # reuse
        buf289 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_146, output_147], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_26.run(buf288, primals_234, primals_226, buf289, 32768, grid=grid(32768), stream=stream0)
        del primals_234
        # Topologically Sorted Source Nodes: [output_148], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, buf55, stride=(1, 1), padding=(0, 16), dilation=(1, 16), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf291 = buf290; del buf290  # reuse
        buf292 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf293 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_148, output_149, add_12, output_151], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_28.run(buf291, primals_236, primals_237, primals_238, primals_239, primals_240, buf279, primals_226, buf292, buf293, 32768, grid=grid(32768), stream=stream0)
        del primals_236
        del primals_240
        # Topologically Sorted Source Nodes: [output_152], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, buf56, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf294, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf295 = buf294; del buf294  # reuse
        buf296 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf297 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [output_152, output_153, output_154], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18.run(buf295, buf297, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, 65536, grid=grid(65536), stream=stream0)
        del primals_242
        # Topologically Sorted Source Nodes: [output_155], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, buf57, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf299 = buf298; del buf298  # reuse
        buf300 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_155, output_156], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf299, primals_249, primals_250, buf300, 65536, grid=grid(65536), stream=stream0)
        del primals_249
        # Topologically Sorted Source Nodes: [output_157], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, buf58, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf302 = buf301; del buf301  # reuse
        buf303 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf304 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [output_157, output_158, output_159], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18.run(buf302, buf304, primals_252, primals_253, primals_254, primals_255, primals_256, primals_250, 65536, grid=grid(65536), stream=stream0)
        del primals_252
        # Topologically Sorted Source Nodes: [output_160], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, buf59, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf306 = buf305; del buf305  # reuse
        buf307 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_160, output_161], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf306, primals_258, primals_250, buf307, 65536, grid=grid(65536), stream=stream0)
        del primals_258
        # Topologically Sorted Source Nodes: [output_162], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, buf60, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf309 = buf308; del buf308  # reuse
        buf310 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf311 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_162, output_163, add_13, output_165], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19.run(buf309, primals_260, primals_261, primals_262, primals_263, primals_264, buf297, primals_250, buf310, buf311, 65536, grid=grid(65536), stream=stream0)
        del primals_260
        del primals_264
        # Topologically Sorted Source Nodes: [output_166], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, buf61, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf313 = buf312; del buf312  # reuse
        buf314 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_166, output_167], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf313, primals_266, primals_267, buf314, 65536, grid=grid(65536), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [output_168], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, buf62, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf316 = buf315; del buf315  # reuse
        buf317 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf318 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [output_168, output_169, output_170], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_18.run(buf316, buf318, primals_269, primals_270, primals_271, primals_272, primals_273, primals_267, 65536, grid=grid(65536), stream=stream0)
        del primals_269
        # Topologically Sorted Source Nodes: [output_171], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, buf63, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf320 = buf319; del buf319  # reuse
        buf321 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_171, output_172], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_17.run(buf320, primals_275, primals_267, buf321, 65536, grid=grid(65536), stream=stream0)
        del primals_275
        # Topologically Sorted Source Nodes: [output_173], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, buf64, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf323 = buf322; del buf322  # reuse
        buf324 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf325 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_173, output_174, add_14, output_176], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_19.run(buf323, primals_277, primals_278, primals_279, primals_280, primals_281, buf311, primals_267, buf324, buf325, 65536, grid=grid(65536), stream=stream0)
        del primals_277
        del primals_281
        # Topologically Sorted Source Nodes: [output_177], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, buf65, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf326, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf327 = buf326; del buf326  # reuse
        buf328 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf329 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [output_177, output_178, output_179], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_29.run(buf327, buf329, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, 65536, grid=grid(65536), stream=stream0)
        del primals_283
        # Topologically Sorted Source Nodes: [output_180], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, buf66, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf331 = buf330; del buf330  # reuse
        buf332 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [output_180, output_181], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_30.run(buf331, primals_290, primals_291, buf332, 65536, grid=grid(65536), stream=stream0)
        del primals_290
        # Topologically Sorted Source Nodes: [output_182], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, buf67, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf334 = buf333; del buf333  # reuse
        buf335 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf336 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [output_182, output_183, output_184], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_29.run(buf334, buf336, primals_293, primals_294, primals_295, primals_296, primals_297, primals_291, 65536, grid=grid(65536), stream=stream0)
        del primals_293
        # Topologically Sorted Source Nodes: [output_185], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, buf68, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf338 = buf337; del buf337  # reuse
        buf339 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [output_185, output_186], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_30.run(buf338, primals_299, primals_291, buf339, 65536, grid=grid(65536), stream=stream0)
        del primals_299
        # Topologically Sorted Source Nodes: [output_187], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, buf69, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf341 = buf340; del buf340  # reuse
        buf342 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf343 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [output_187, output_188, add_15, output_190], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_31.run(buf341, primals_301, primals_302, primals_303, primals_304, primals_305, buf329, primals_291, buf342, buf343, 65536, grid=grid(65536), stream=stream0)
        del primals_301
        del primals_305
        # Topologically Sorted Source Nodes: [output_191], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, buf70, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf345 = buf344; del buf344  # reuse
        buf346 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [output_191, output_192], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_30.run(buf345, primals_307, primals_308, buf346, 65536, grid=grid(65536), stream=stream0)
        del primals_307
        # Topologically Sorted Source Nodes: [output_193], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, buf71, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf348 = buf347; del buf347  # reuse
        buf349 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf350 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [output_193, output_194, output_195], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_29.run(buf348, buf350, primals_310, primals_311, primals_312, primals_313, primals_314, primals_308, 65536, grid=grid(65536), stream=stream0)
        del primals_310
        # Topologically Sorted Source Nodes: [output_196], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, buf72, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf352 = buf351; del buf351  # reuse
        buf353 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [output_196, output_197], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_30.run(buf352, primals_316, primals_308, buf353, 65536, grid=grid(65536), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [output_198], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, buf73, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf355 = buf354; del buf354  # reuse
        buf356 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf357 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_198, output_199, add_16, output_201], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_32.run(buf355, primals_318, primals_319, primals_320, primals_321, primals_322, buf343, primals_308, buf356, buf357, 4096, 16, grid=grid(4096, 16), stream=stream0)
        del primals_318
        del primals_322
    return (buf357, buf0, buf1, primals_3, primals_4, primals_5, primals_6, primals_7, buf2, primals_9, primals_10, primals_11, primals_12, primals_13, buf3, primals_16, buf4, primals_19, primals_20, primals_21, primals_22, buf5, buf6, primals_27, primals_28, primals_29, buf7, primals_33, buf8, primals_36, primals_37, primals_38, primals_39, buf9, buf10, primals_44, primals_45, primals_46, buf11, primals_50, buf12, primals_53, primals_54, primals_55, primals_56, buf13, buf14, primals_61, primals_62, primals_63, buf15, primals_67, buf16, primals_70, primals_71, primals_72, primals_73, buf17, buf18, primals_78, primals_79, primals_80, buf19, primals_84, buf20, primals_87, primals_88, primals_89, primals_90, buf21, buf22, primals_95, primals_96, primals_97, buf23, primals_100, primals_101, primals_102, primals_103, primals_104, buf24, primals_107, buf25, primals_110, primals_111, primals_112, primals_113, buf26, buf27, primals_118, primals_119, primals_120, buf28, primals_124, buf29, primals_127, primals_128, primals_129, primals_130, buf30, buf31, primals_135, primals_136, primals_137, buf32, primals_141, buf33, primals_144, primals_145, primals_146, primals_147, buf34, buf35, primals_152, primals_153, primals_154, buf36, primals_158, buf37, primals_161, primals_162, primals_163, primals_164, buf38, buf39, primals_169, primals_170, primals_171, buf40, primals_175, buf41, primals_178, primals_179, primals_180, primals_181, buf42, buf43, primals_186, primals_187, primals_188, buf44, primals_192, buf45, primals_195, primals_196, primals_197, primals_198, buf46, buf47, primals_203, primals_204, primals_205, buf48, primals_209, buf49, primals_212, primals_213, primals_214, primals_215, buf50, buf51, primals_220, primals_221, primals_222, buf52, primals_226, buf53, primals_229, primals_230, primals_231, primals_232, buf54, buf55, primals_237, primals_238, primals_239, buf56, primals_243, primals_244, primals_245, primals_246, primals_247, buf57, primals_250, buf58, primals_253, primals_254, primals_255, primals_256, buf59, buf60, primals_261, primals_262, primals_263, buf61, primals_267, buf62, primals_270, primals_271, primals_272, primals_273, buf63, buf64, primals_278, primals_279, primals_280, buf65, primals_284, primals_285, primals_286, primals_287, primals_288, buf66, primals_291, buf67, primals_294, primals_295, primals_296, primals_297, buf68, buf69, primals_302, primals_303, primals_304, buf70, primals_308, buf71, primals_311, primals_312, primals_313, primals_314, buf72, buf73, primals_319, primals_320, primals_321, buf79, buf81, buf83, buf84, buf85, buf86, buf87, buf90, buf92, buf94, buf96, buf98, buf99, buf101, buf103, buf105, buf106, buf108, buf109, buf110, buf112, buf113, buf115, buf117, buf119, buf120, buf122, buf123, buf124, buf126, buf127, buf129, buf131, buf133, buf134, buf136, buf137, buf138, buf140, buf141, buf143, buf145, buf147, buf148, buf150, buf151, buf152, buf154, buf155, buf157, buf159, buf161, buf162, buf164, buf165, buf166, buf168, buf169, buf170, buf171, buf172, buf175, buf177, buf179, buf181, buf183, buf184, buf186, buf188, buf190, buf191, buf193, buf194, buf195, buf197, buf198, buf200, buf202, buf204, buf205, buf207, buf208, buf209, buf211, buf212, buf214, buf216, buf218, buf219, buf221, buf222, buf223, buf225, buf226, buf228, buf230, buf232, buf233, buf235, buf236, buf237, buf239, buf240, buf242, buf244, buf246, buf247, buf249, buf250, buf251, buf253, buf254, buf256, buf258, buf260, buf261, buf263, buf264, buf265, buf267, buf268, buf270, buf272, buf274, buf275, buf277, buf278, buf279, buf281, buf282, buf284, buf286, buf288, buf289, buf291, buf292, buf293, buf295, buf297, buf299, buf300, buf302, buf304, buf306, buf307, buf309, buf310, buf311, buf313, buf314, buf316, buf318, buf320, buf321, buf323, buf324, buf325, buf327, buf329, buf331, buf332, buf334, buf336, buf338, buf339, buf341, buf342, buf343, buf345, buf346, buf348, buf350, buf352, buf353, buf355, buf356, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((13, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((48, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((128, 128, 3, 1), (384, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, 128, 1, 3), (384, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
