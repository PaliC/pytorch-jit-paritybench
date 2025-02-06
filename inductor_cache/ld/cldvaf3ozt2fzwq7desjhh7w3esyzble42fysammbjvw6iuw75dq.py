# AOT ID: ['16_forward']
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


# kernel path: inductor_cache/hg/chgkd3ydi7fhziw6npgs2c67hmzzb4huslntxunkkvg3edpr5ed2.py
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
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 64*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ut/cut5gf5f67imxk5hrq2hrndzi4kuqloragwlibs36radlx3f4pze.py
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
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/xr/cxrshjxzhazcwkmapyigahnzwcbojl2nsysnz5lwe7spegfc6j6g.py
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
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 128*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vp/cvpounvh4nxwr3murlvhn3hslencyhl7ukyo6x6y4l4om2zujvsd.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ns/cns5fvqtkydpgnnaki5gpnslxdphmn2g3ozfg6zd7gmlc5zfbkdd.py
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
    size_hints={'y': 1024, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 5
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
    tmp0 = tl.load(in_ptr0 + (x2 + 5*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 160*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p4/cp4s5fwrw6tls4jhxfhrapqv3wxf3ktniboge5aqi2kjuxc55ijj.py
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
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 4*x2 + 16*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/3c/c3caicljjugtlkiakdpmazwzbtsdn5pgbotu7r6srlo5rbf557cl.py
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
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 4*x2 + 36*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jp/cjpcuwb2alhscmdoqjbhzf4w6rh5dq42g3sxdmdnqmpz3grdblka.py
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
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 4*x2 + 16*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ol/colh2mnjsgodft76d6d4kcv4eaev52ws44riozuoxgkie7zuhz67.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   x => cat
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => gt, mul_3, where
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %getitem], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.22916666666666666), kwargs = {})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_rrelu_with_noise_functional_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_rrelu_with_noise_functional_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_rrelu_with_noise_functional_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_rrelu_with_noise_functional_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x3 = xindex // 16
    x1 = ((xindex // 16) % 32)
    x2 = xindex // 512
    x4 = xindex
    tmp19 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 13, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (13*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 16, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (6*x1 + 384*x2 + ((-13) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr1 + (3 + 6*x1 + 384*x2 + ((-13) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.load(in_ptr1 + (192 + 6*x1 + 384*x2 + ((-13) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.load(in_ptr1 + (195 + 6*x1 + 384*x2 + ((-13) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp5, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tl.full([1], 1, tl.int32)
    tmp26 = tmp25 / tmp24
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = 0.0
    tmp35 = tmp33 > tmp34
    tmp36 = 0.22916666666666666
    tmp37 = tmp33 * tmp36
    tmp38 = tl.where(tmp35, tmp33, tmp37)
    tl.store(out_ptr0 + (x4), tmp18, None)
    tl.store(in_out_ptr0 + (x4), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/pu/cpuxyrstytnbfv3kbzmkyam4oxhtw2ffshlyk5afoup43nosv7er.py
# Topologically Sorted Source Nodes: [max_pool2d_1], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   max_pool2d_1 => _low_memory_max_pool2d_offsets_to_indices_1, _low_memory_max_pool2d_with_offsets_1, getitem_2
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%where, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_2 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %_low_memory_max_pool2d_offsets_to_indices_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default](args = (%getitem_3, 2, 32, [2, 2], [0, 0]), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_11(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    x2 = xindex // 256
    x5 = xindex
    x3 = ((xindex // 256) % 16)
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x1 + 1024*x2), None)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 32*x1 + 1024*x2), None)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 32*x1 + 1024*x2), None)
    tmp5 = tl.load(in_ptr0 + (528 + x0 + 32*x1 + 1024*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tmp17 = tl.full([1], 2, tl.int32)
    tmp18 = tl.where((tmp16 < 0) != (tmp17 < 0), tl.where(tmp16 % tmp17 != 0, tmp16 // tmp17 - 1, tmp16 // tmp17), tmp16 // tmp17)
    tmp19 = tmp18 * tmp17
    tmp20 = tmp16 - tmp19
    tmp21 = 2*x3
    tmp22 = tmp21 + tmp18
    tmp23 = 2*x1
    tmp24 = tmp23 + tmp20
    tmp25 = tl.full([1], 32, tl.int64)
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tl.store(out_ptr0 + (x5), tmp6, None)
    tl.store(out_ptr1 + (x5), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/2m/c2mgqavoks22zp2ljd3vbkst4cuur25jswkyid6gnwxfy6frc7pe.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   input_4 => add_5, mul_8, mul_9, sub_2
#   input_5 => gt_1, mul_10, where_1
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %unsqueeze_23), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_5, 0), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.22916666666666666), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_5, %mul_10), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.22916666666666666
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/vp/cvp7crbcb5axaxjdrmeyynb7debluaugywkw2osaxsayxc4gyqbm.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   input_7 => add_7, mul_12, mul_13, sub_3
#   input_8 => gt_2, mul_14, where_2
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_31), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.22916666666666666), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_7, %mul_14), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.22916666666666666
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/fo/cfo3wae5o6wfg34shmmskevnbnlbmnugbnfagj5bs4clrn3x5veo.py
# Topologically Sorted Source Nodes: [input_2, input_10, add, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   add => add_10
#   input_10 => add_9, mul_16, mul_17, sub_4
#   input_2 => add_3, mul_5, mul_6, sub_1
#   out => gt_3, mul_18, where_3
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_39), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %add_3), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_10, 0), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.22916666666666666), kwargs = {})
#   %where_3 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_10, %mul_18), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = 0.0
    tmp31 = tmp29 > tmp30
    tmp32 = 0.22916666666666666
    tmp33 = tmp29 * tmp32
    tmp34 = tl.where(tmp31, tmp29, tmp33)
    tl.store(in_out_ptr0 + (x2), tmp34, None)
''', device_str='cuda')


# kernel path: inductor_cache/ln/clnypebitkhhg54wprl3gjzxd4l5e5ydm6pskdozkd22x6p5ddes.py
# Topologically Sorted Source Nodes: [input_19, add_1, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   add_1 => add_17
#   input_19 => add_16, mul_28, mul_29, sub_7
#   out_1 => gt_6, mul_30, where_6
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_61), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_63), kwargs = {})
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %where_3), kwargs = {})
#   %gt_6 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_17, 0), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, 0.22916666666666666), kwargs = {})
#   %where_6 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_17, %mul_30), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.22916666666666666
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/t7/ct7wskolahkujyxjft7oyiun5gqgwaazqciwcdxh2zobnhvdqdmr.py
# Topologically Sorted Source Nodes: [max_pool2d_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   max_pool2d_2 => _low_memory_max_pool2d_offsets_to_indices_2, _low_memory_max_pool2d_with_offsets_2, getitem_4
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%where_33, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_4 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
#   %_low_memory_max_pool2d_offsets_to_indices_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default](args = (%getitem_5, 2, 16, [2, 2], [0, 0]), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_16 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_16(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 8)
    x2 = xindex // 512
    x5 = xindex
    x3 = ((xindex // 512) % 8)
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 2048*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 2048*x2), None)
    tmp3 = tl.load(in_ptr0 + (1024 + x0 + 128*x1 + 2048*x2), None)
    tmp5 = tl.load(in_ptr0 + (1088 + x0 + 128*x1 + 2048*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tmp17 = tl.full([1], 2, tl.int32)
    tmp18 = tl.where((tmp16 < 0) != (tmp17 < 0), tl.where(tmp16 % tmp17 != 0, tmp16 // tmp17 - 1, tmp16 // tmp17), tmp16 // tmp17)
    tmp19 = tmp18 * tmp17
    tmp20 = tmp16 - tmp19
    tmp21 = 2*x3
    tmp22 = tmp21 + tmp18
    tmp23 = 2*x1
    tmp24 = tmp23 + tmp20
    tmp25 = tl.full([1], 16, tl.int64)
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tl.store(out_ptr0 + (x5), tmp6, None)
    tl.store(out_ptr1 + (x5), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/hk/chkwtgi4d3vqjkxo2udg7so5zwubcy3hfoc7usjkwbfdjxbwmw27.py
# Topologically Sorted Source Nodes: [input_105, input_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   input_105 => add_84, mul_143, mul_144, sub_36
#   input_106 => gt_34, mul_145, where_34
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_289), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_143, %unsqueeze_293), kwargs = {})
#   %add_84 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_144, %unsqueeze_295), kwargs = {})
#   %gt_34 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_84, 0), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_84, 0.22916666666666666), kwargs = {})
#   %where_34 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_34, %add_84, %mul_145), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.22916666666666666
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/tx/ctxntouzvyvo5gbefd4cfo4wdlkdd4fkx2dojwyobbneisahtn3c.py
# Topologically Sorted Source Nodes: [input_108, input_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   input_108 => add_86, mul_147, mul_148, sub_37
#   input_109 => gt_35, mul_149, where_35
# Graph fragment:
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_297), kwargs = {})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_147, %unsqueeze_301), kwargs = {})
#   %add_86 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_148, %unsqueeze_303), kwargs = {})
#   %gt_35 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_86, 0), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_86, 0.22916666666666666), kwargs = {})
#   %where_35 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_35, %add_86, %mul_149), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.22916666666666666
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/oh/cohhf7yggde6ihunlgwmyxexv5nkufeyfjd4whp2fhtzsu6nzekq.py
# Topologically Sorted Source Nodes: [input_103, input_111, add_11, out_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   add_11 => add_89
#   input_103 => add_82, mul_140, mul_141, sub_35
#   input_111 => add_88, mul_151, mul_152, sub_38
#   out_11 => gt_36, mul_153, where_36
# Graph fragment:
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_281), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_140, %unsqueeze_285), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_141, %unsqueeze_287), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_151, %unsqueeze_309), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_152, %unsqueeze_311), kwargs = {})
#   %add_89 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %add_82), kwargs = {})
#   %gt_36 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_89, 0), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, 0.22916666666666666), kwargs = {})
#   %where_36 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_36, %add_89, %mul_153), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = 0.0
    tmp31 = tmp29 > tmp30
    tmp32 = 0.22916666666666666
    tmp33 = tmp29 * tmp32
    tmp34 = tl.where(tmp31, tmp29, tmp33)
    tl.store(in_out_ptr0 + (x2), tmp34, None)
''', device_str='cuda')


# kernel path: inductor_cache/6o/c6oynsi3nypnfitvejbyl2ryus2fyjfqtlwfkk5exur3ffe6suvt.py
# Topologically Sorted Source Nodes: [input_120, add_12, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   add_12 => add_96
#   input_120 => add_95, mul_163, mul_164, sub_41
#   out_12 => gt_39, mul_165, where_39
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %unsqueeze_333), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %unsqueeze_335), kwargs = {})
#   %add_96 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_95, %where_36), kwargs = {})
#   %gt_39 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_96, 0), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_96, 0.22916666666666666), kwargs = {})
#   %where_39 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_39, %add_96, %mul_165), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.22916666666666666
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/hs/chsk5imy6zuhfyse6ymjqdmbdl37hm442lurh7m7sbpbtg73og32.py
# Topologically Sorted Source Nodes: [out_up], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   out_up => full_default, index_put
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([65536], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%view_1], %view_3), kwargs = {})
triton_poi_fused_max_unpool2d_21 = async_compile.triton('triton_poi_fused_max_unpool2d_21', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_unpool2d_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_unpool2d_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/lb/clba6enbylwu75fns2z576v77njhz2xfenc2py6vvoypbf5ysxyc.py
# Topologically Sorted Source Nodes: [out_up], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   out_up => full_default, index_put
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([65536], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%view_1], %view_3), kwargs = {})
triton_poi_fused_max_unpool2d_22 = async_compile.triton('triton_poi_fused_max_unpool2d_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_unpool2d_22', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_unpool2d_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (64*((x0 % 64)) + 4096*(x0 // 4096) + (((x0 // 64) % 64))), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (64*((x0 % 64)) + 4096*(x0 // 4096) + (((x0 // 64) % 64))), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (((x0 // 64) % 64)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (((x0 // 64) % 64)), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (((x0 // 64) % 64)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (((x0 // 64) % 64)), None, eviction_policy='evict_last')
    tmp1 = 256*(x0 // 64)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([XBLOCK], 65536, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert((0 <= tmp6) & (tmp6 < 65536), "index out of bounds: 0 <= tmp6 < 65536")
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.store(out_ptr0 + (tl.broadcast_to(tmp6, [XBLOCK])), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/uj/cuj3gjsytye3e6n5i2odbb3hxnm2m54zjjqh2bmsj5xkieq227kh.py
# Topologically Sorted Source Nodes: [input_309, input_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   input_309 => add_241, mul_411, mul_412, sub_103
#   input_310 => gt_100, mul_413, where_100
# Graph fragment:
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_825), kwargs = {})
#   %mul_411 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %mul_412 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_411, %unsqueeze_829), kwargs = {})
#   %add_241 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_412, %unsqueeze_831), kwargs = {})
#   %gt_100 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_241, 0), kwargs = {})
#   %mul_413 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_241, 0.22916666666666666), kwargs = {})
#   %where_100 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_100, %add_241, %mul_413), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.22916666666666666
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/s4/cs4huljedcbbnwcyftnr7a5vqbtraeeg5nqflqc6jyzr5emvn3gt.py
# Topologically Sorted Source Nodes: [input_315, add_33, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   add_33 => add_246
#   input_315 => add_245, mul_419, mul_420, sub_105
#   out_33 => gt_102, mul_421, where_102
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_109, %unsqueeze_841), kwargs = {})
#   %mul_419 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_420 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_419, %unsqueeze_845), kwargs = {})
#   %add_245 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_420, %unsqueeze_847), kwargs = {})
#   %add_246 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %add_245), kwargs = {})
#   %gt_102 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_246, 0), kwargs = {})
#   %mul_421 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_246, 0.22916666666666666), kwargs = {})
#   %where_102 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_102, %add_246, %mul_421), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 64*x2 + 16384*y1), xmask & ymask)
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.22916666666666666
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(out_ptr1 + (y0 + 64*x2 + 16384*y1), tmp19, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 64*x2 + 16384*y1), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zq/czqahx2y26idjhgaufszs2csaonm6txy6uffyrkdrktwjd3umxda.py
# Topologically Sorted Source Nodes: [out_up, out_up_1], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   out_up => full_default
#   out_up_1 => index_put_1
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([65536], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%view_6], %view_8), kwargs = {})
triton_poi_fused_max_unpool2d_25 = async_compile.triton('triton_poi_fused_max_unpool2d_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_unpool2d_25', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_unpool2d_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*((x0 % 256)) + 4096*(x0 // 4096) + (((x0 // 256) % 16))), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (16*((x0 % 256)) + 4096*(x0 // 4096) + (((x0 // 256) % 16))), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (((x0 // 256) % 16)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (((x0 // 256) % 16)), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (((x0 // 256) % 16)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (((x0 // 256) % 16)), None, eviction_policy='evict_last')
    tmp1 = 1024*(x0 // 256)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([XBLOCK], 65536, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert((0 <= tmp6) & (tmp6 < 65536), "index out of bounds: 0 <= tmp6 < 65536")
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.store(out_ptr0 + (tl.broadcast_to(tmp6, [XBLOCK])), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/6d/c6dq3m5x6nbyche72lped2ti6c42rz6d2npoffhhy7mdvpgd3sek.py
# Topologically Sorted Source Nodes: [input_410, input_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   input_410 => add_321, mul_547, mul_548, sub_137
#   input_411 => gt_133, mul_549, where_133
# Graph fragment:
#   %sub_137 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_141, %unsqueeze_1097), kwargs = {})
#   %mul_547 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_137, %unsqueeze_1099), kwargs = {})
#   %mul_548 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_547, %unsqueeze_1101), kwargs = {})
#   %add_321 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_548, %unsqueeze_1103), kwargs = {})
#   %gt_133 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_321, 0), kwargs = {})
#   %mul_549 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_321, 0.22916666666666666), kwargs = {})
#   %where_133 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_133, %add_321, %mul_549), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4)
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
    tmp18 = 0.22916666666666666
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/iu/ciuh2igw6cyttwsqqjpmio4wh6eu72etwy4zrf75m2hbcyg4ymli.py
# Topologically Sorted Source Nodes: [input_413, input_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   input_413 => add_323, mul_551, mul_552, sub_138
#   input_414 => gt_134, mul_553, where_134
# Graph fragment:
#   %sub_138 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_142, %unsqueeze_1105), kwargs = {})
#   %mul_551 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_138, %unsqueeze_1107), kwargs = {})
#   %mul_552 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_551, %unsqueeze_1109), kwargs = {})
#   %add_323 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_552, %unsqueeze_1111), kwargs = {})
#   %gt_134 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_323, 0), kwargs = {})
#   %mul_553 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_323, 0.22916666666666666), kwargs = {})
#   %where_134 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_134, %add_323, %mul_553), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4)
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
    tmp18 = 0.22916666666666666
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/fz/cfzqgjunbhmfgdemy2glwwzqcvvwx2zhzpz67cd5cd3cig4h2vci.py
# Topologically Sorted Source Nodes: [input_416, add_44, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   add_44 => add_326
#   input_416 => add_325, mul_555, mul_556, sub_139
#   out_44 => gt_135, mul_557, where_135
# Graph fragment:
#   %sub_139 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_143, %unsqueeze_1113), kwargs = {})
#   %mul_555 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_139, %unsqueeze_1115), kwargs = {})
#   %mul_556 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_555, %unsqueeze_1117), kwargs = {})
#   %add_325 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_556, %unsqueeze_1119), kwargs = {})
#   %add_326 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %add_325), kwargs = {})
#   %gt_135 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_326, 0), kwargs = {})
#   %mul_557 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_326, 0.22916666666666666), kwargs = {})
#   %where_135 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_135, %add_326, %mul_557), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 16*x2 + 16384*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.22916666666666666
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(out_ptr1 + (y0 + 16*x2 + 16384*y1), tmp19, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 16*x2 + 16384*y1), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ct/cctshsmolypxapuufrpsopv4nkkh5gu55pebo2hwvtpkf364x3zo.py
# Topologically Sorted Source Nodes: [input_425, add_45, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   add_45 => add_333
#   input_425 => add_332, mul_567, mul_568, sub_142
#   out_45 => gt_138, mul_569, where_138
# Graph fragment:
#   %sub_142 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_146, %unsqueeze_1137), kwargs = {})
#   %mul_567 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_142, %unsqueeze_1139), kwargs = {})
#   %mul_568 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_567, %unsqueeze_1141), kwargs = {})
#   %add_332 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_568, %unsqueeze_1143), kwargs = {})
#   %add_333 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_332, %where_135), kwargs = {})
#   %gt_138 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_333, 0), kwargs = {})
#   %mul_569 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_333, 0.22916666666666666), kwargs = {})
#   %where_138 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_138, %add_333, %mul_569), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.22916666666666666
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/r4/cr4lwss6cuc3q6uce23wgpmso5kuqs4ojgwt6cp2zvqalvenfevt.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_3 => convolution_174
# Graph fragment:
#   %convolution_174 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_165, %primals_856, None, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_30 = async_compile.triton('triton_poi_fused_convolution_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16384*y1), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp0, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856 = args
    args.clear()
    assert_size_stride(primals_1, (13, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_13, (16, ), (1, ))
    assert_size_stride(primals_14, (16, ), (1, ))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (16, ), (1, ))
    assert_size_stride(primals_17, (16, 16, 2, 2), (64, 4, 2, 1))
    assert_size_stride(primals_18, (16, ), (1, ))
    assert_size_stride(primals_19, (16, ), (1, ))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (16, ), (1, ))
    assert_size_stride(primals_22, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_28, (16, ), (1, ))
    assert_size_stride(primals_29, (16, ), (1, ))
    assert_size_stride(primals_30, (16, ), (1, ))
    assert_size_stride(primals_31, (16, ), (1, ))
    assert_size_stride(primals_32, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_33, (16, ), (1, ))
    assert_size_stride(primals_34, (16, ), (1, ))
    assert_size_stride(primals_35, (16, ), (1, ))
    assert_size_stride(primals_36, (16, ), (1, ))
    assert_size_stride(primals_37, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_43, (16, ), (1, ))
    assert_size_stride(primals_44, (16, ), (1, ))
    assert_size_stride(primals_45, (16, ), (1, ))
    assert_size_stride(primals_46, (16, ), (1, ))
    assert_size_stride(primals_47, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_48, (16, ), (1, ))
    assert_size_stride(primals_49, (16, ), (1, ))
    assert_size_stride(primals_50, (16, ), (1, ))
    assert_size_stride(primals_51, (16, ), (1, ))
    assert_size_stride(primals_52, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_58, (16, ), (1, ))
    assert_size_stride(primals_59, (16, ), (1, ))
    assert_size_stride(primals_60, (16, ), (1, ))
    assert_size_stride(primals_61, (16, ), (1, ))
    assert_size_stride(primals_62, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_63, (16, ), (1, ))
    assert_size_stride(primals_64, (16, ), (1, ))
    assert_size_stride(primals_65, (16, ), (1, ))
    assert_size_stride(primals_66, (16, ), (1, ))
    assert_size_stride(primals_67, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (16, ), (1, ))
    assert_size_stride(primals_74, (16, ), (1, ))
    assert_size_stride(primals_75, (16, ), (1, ))
    assert_size_stride(primals_76, (16, ), (1, ))
    assert_size_stride(primals_77, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_78, (16, ), (1, ))
    assert_size_stride(primals_79, (16, ), (1, ))
    assert_size_stride(primals_80, (16, ), (1, ))
    assert_size_stride(primals_81, (16, ), (1, ))
    assert_size_stride(primals_82, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (64, ), (1, ))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (64, ), (1, ))
    assert_size_stride(primals_87, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_88, (16, ), (1, ))
    assert_size_stride(primals_89, (16, ), (1, ))
    assert_size_stride(primals_90, (16, ), (1, ))
    assert_size_stride(primals_91, (16, ), (1, ))
    assert_size_stride(primals_92, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_93, (16, ), (1, ))
    assert_size_stride(primals_94, (16, ), (1, ))
    assert_size_stride(primals_95, (16, ), (1, ))
    assert_size_stride(primals_96, (16, ), (1, ))
    assert_size_stride(primals_97, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_98, (64, ), (1, ))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (64, ), (1, ))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_103, (16, ), (1, ))
    assert_size_stride(primals_104, (16, ), (1, ))
    assert_size_stride(primals_105, (16, ), (1, ))
    assert_size_stride(primals_106, (16, ), (1, ))
    assert_size_stride(primals_107, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_108, (16, ), (1, ))
    assert_size_stride(primals_109, (16, ), (1, ))
    assert_size_stride(primals_110, (16, ), (1, ))
    assert_size_stride(primals_111, (16, ), (1, ))
    assert_size_stride(primals_112, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_113, (64, ), (1, ))
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_116, (64, ), (1, ))
    assert_size_stride(primals_117, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_118, (16, ), (1, ))
    assert_size_stride(primals_119, (16, ), (1, ))
    assert_size_stride(primals_120, (16, ), (1, ))
    assert_size_stride(primals_121, (16, ), (1, ))
    assert_size_stride(primals_122, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_123, (16, ), (1, ))
    assert_size_stride(primals_124, (16, ), (1, ))
    assert_size_stride(primals_125, (16, ), (1, ))
    assert_size_stride(primals_126, (16, ), (1, ))
    assert_size_stride(primals_127, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_128, (64, ), (1, ))
    assert_size_stride(primals_129, (64, ), (1, ))
    assert_size_stride(primals_130, (64, ), (1, ))
    assert_size_stride(primals_131, (64, ), (1, ))
    assert_size_stride(primals_132, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_133, (16, ), (1, ))
    assert_size_stride(primals_134, (16, ), (1, ))
    assert_size_stride(primals_135, (16, ), (1, ))
    assert_size_stride(primals_136, (16, ), (1, ))
    assert_size_stride(primals_137, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_138, (16, ), (1, ))
    assert_size_stride(primals_139, (16, ), (1, ))
    assert_size_stride(primals_140, (16, ), (1, ))
    assert_size_stride(primals_141, (16, ), (1, ))
    assert_size_stride(primals_142, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_143, (64, ), (1, ))
    assert_size_stride(primals_144, (64, ), (1, ))
    assert_size_stride(primals_145, (64, ), (1, ))
    assert_size_stride(primals_146, (64, ), (1, ))
    assert_size_stride(primals_147, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_148, (16, ), (1, ))
    assert_size_stride(primals_149, (16, ), (1, ))
    assert_size_stride(primals_150, (16, ), (1, ))
    assert_size_stride(primals_151, (16, ), (1, ))
    assert_size_stride(primals_152, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_153, (16, ), (1, ))
    assert_size_stride(primals_154, (16, ), (1, ))
    assert_size_stride(primals_155, (16, ), (1, ))
    assert_size_stride(primals_156, (16, ), (1, ))
    assert_size_stride(primals_157, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_158, (64, ), (1, ))
    assert_size_stride(primals_159, (64, ), (1, ))
    assert_size_stride(primals_160, (64, ), (1, ))
    assert_size_stride(primals_161, (64, ), (1, ))
    assert_size_stride(primals_162, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_163, (16, ), (1, ))
    assert_size_stride(primals_164, (16, ), (1, ))
    assert_size_stride(primals_165, (16, ), (1, ))
    assert_size_stride(primals_166, (16, ), (1, ))
    assert_size_stride(primals_167, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_168, (16, ), (1, ))
    assert_size_stride(primals_169, (16, ), (1, ))
    assert_size_stride(primals_170, (16, ), (1, ))
    assert_size_stride(primals_171, (16, ), (1, ))
    assert_size_stride(primals_172, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, ), (1, ))
    assert_size_stride(primals_175, (64, ), (1, ))
    assert_size_stride(primals_176, (64, ), (1, ))
    assert_size_stride(primals_177, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_183, (32, ), (1, ))
    assert_size_stride(primals_184, (32, ), (1, ))
    assert_size_stride(primals_185, (32, ), (1, ))
    assert_size_stride(primals_186, (32, ), (1, ))
    assert_size_stride(primals_187, (32, 32, 2, 2), (128, 4, 2, 1))
    assert_size_stride(primals_188, (32, ), (1, ))
    assert_size_stride(primals_189, (32, ), (1, ))
    assert_size_stride(primals_190, (32, ), (1, ))
    assert_size_stride(primals_191, (32, ), (1, ))
    assert_size_stride(primals_192, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_193, (128, ), (1, ))
    assert_size_stride(primals_194, (128, ), (1, ))
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_198, (32, ), (1, ))
    assert_size_stride(primals_199, (32, ), (1, ))
    assert_size_stride(primals_200, (32, ), (1, ))
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_203, (32, ), (1, ))
    assert_size_stride(primals_204, (32, ), (1, ))
    assert_size_stride(primals_205, (32, ), (1, ))
    assert_size_stride(primals_206, (32, ), (1, ))
    assert_size_stride(primals_207, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_208, (128, ), (1, ))
    assert_size_stride(primals_209, (128, ), (1, ))
    assert_size_stride(primals_210, (128, ), (1, ))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_213, (32, ), (1, ))
    assert_size_stride(primals_214, (32, ), (1, ))
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (32, ), (1, ))
    assert_size_stride(primals_217, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_218, (32, ), (1, ))
    assert_size_stride(primals_219, (32, ), (1, ))
    assert_size_stride(primals_220, (32, ), (1, ))
    assert_size_stride(primals_221, (32, ), (1, ))
    assert_size_stride(primals_222, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_223, (128, ), (1, ))
    assert_size_stride(primals_224, (128, ), (1, ))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_227, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_228, (32, ), (1, ))
    assert_size_stride(primals_229, (32, ), (1, ))
    assert_size_stride(primals_230, (32, ), (1, ))
    assert_size_stride(primals_231, (32, ), (1, ))
    assert_size_stride(primals_232, (32, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_233, (32, 32, 1, 5), (160, 5, 5, 1))
    assert_size_stride(primals_234, (32, ), (1, ))
    assert_size_stride(primals_235, (32, ), (1, ))
    assert_size_stride(primals_236, (32, ), (1, ))
    assert_size_stride(primals_237, (32, ), (1, ))
    assert_size_stride(primals_238, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_239, (128, ), (1, ))
    assert_size_stride(primals_240, (128, ), (1, ))
    assert_size_stride(primals_241, (128, ), (1, ))
    assert_size_stride(primals_242, (128, ), (1, ))
    assert_size_stride(primals_243, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_244, (32, ), (1, ))
    assert_size_stride(primals_245, (32, ), (1, ))
    assert_size_stride(primals_246, (32, ), (1, ))
    assert_size_stride(primals_247, (32, ), (1, ))
    assert_size_stride(primals_248, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_249, (32, ), (1, ))
    assert_size_stride(primals_250, (32, ), (1, ))
    assert_size_stride(primals_251, (32, ), (1, ))
    assert_size_stride(primals_252, (32, ), (1, ))
    assert_size_stride(primals_253, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_254, (128, ), (1, ))
    assert_size_stride(primals_255, (128, ), (1, ))
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_259, (32, ), (1, ))
    assert_size_stride(primals_260, (32, ), (1, ))
    assert_size_stride(primals_261, (32, ), (1, ))
    assert_size_stride(primals_262, (32, ), (1, ))
    assert_size_stride(primals_263, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_264, (32, ), (1, ))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, ), (1, ))
    assert_size_stride(primals_267, (32, ), (1, ))
    assert_size_stride(primals_268, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (128, ), (1, ))
    assert_size_stride(primals_271, (128, ), (1, ))
    assert_size_stride(primals_272, (128, ), (1, ))
    assert_size_stride(primals_273, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_274, (32, ), (1, ))
    assert_size_stride(primals_275, (32, ), (1, ))
    assert_size_stride(primals_276, (32, ), (1, ))
    assert_size_stride(primals_277, (32, ), (1, ))
    assert_size_stride(primals_278, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_279, (32, ), (1, ))
    assert_size_stride(primals_280, (32, ), (1, ))
    assert_size_stride(primals_281, (32, ), (1, ))
    assert_size_stride(primals_282, (32, ), (1, ))
    assert_size_stride(primals_283, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_284, (128, ), (1, ))
    assert_size_stride(primals_285, (128, ), (1, ))
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_288, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_289, (32, ), (1, ))
    assert_size_stride(primals_290, (32, ), (1, ))
    assert_size_stride(primals_291, (32, ), (1, ))
    assert_size_stride(primals_292, (32, ), (1, ))
    assert_size_stride(primals_293, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_294, (32, ), (1, ))
    assert_size_stride(primals_295, (32, ), (1, ))
    assert_size_stride(primals_296, (32, ), (1, ))
    assert_size_stride(primals_297, (32, ), (1, ))
    assert_size_stride(primals_298, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_299, (128, ), (1, ))
    assert_size_stride(primals_300, (128, ), (1, ))
    assert_size_stride(primals_301, (128, ), (1, ))
    assert_size_stride(primals_302, (128, ), (1, ))
    assert_size_stride(primals_303, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_304, (32, ), (1, ))
    assert_size_stride(primals_305, (32, ), (1, ))
    assert_size_stride(primals_306, (32, ), (1, ))
    assert_size_stride(primals_307, (32, ), (1, ))
    assert_size_stride(primals_308, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_309, (32, ), (1, ))
    assert_size_stride(primals_310, (32, ), (1, ))
    assert_size_stride(primals_311, (32, ), (1, ))
    assert_size_stride(primals_312, (32, ), (1, ))
    assert_size_stride(primals_313, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_314, (128, ), (1, ))
    assert_size_stride(primals_315, (128, ), (1, ))
    assert_size_stride(primals_316, (128, ), (1, ))
    assert_size_stride(primals_317, (128, ), (1, ))
    assert_size_stride(primals_318, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_319, (32, ), (1, ))
    assert_size_stride(primals_320, (32, ), (1, ))
    assert_size_stride(primals_321, (32, ), (1, ))
    assert_size_stride(primals_322, (32, ), (1, ))
    assert_size_stride(primals_323, (32, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_324, (32, 32, 1, 5), (160, 5, 5, 1))
    assert_size_stride(primals_325, (32, ), (1, ))
    assert_size_stride(primals_326, (32, ), (1, ))
    assert_size_stride(primals_327, (32, ), (1, ))
    assert_size_stride(primals_328, (32, ), (1, ))
    assert_size_stride(primals_329, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_330, (128, ), (1, ))
    assert_size_stride(primals_331, (128, ), (1, ))
    assert_size_stride(primals_332, (128, ), (1, ))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_335, (32, ), (1, ))
    assert_size_stride(primals_336, (32, ), (1, ))
    assert_size_stride(primals_337, (32, ), (1, ))
    assert_size_stride(primals_338, (32, ), (1, ))
    assert_size_stride(primals_339, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_340, (32, ), (1, ))
    assert_size_stride(primals_341, (32, ), (1, ))
    assert_size_stride(primals_342, (32, ), (1, ))
    assert_size_stride(primals_343, (32, ), (1, ))
    assert_size_stride(primals_344, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_345, (128, ), (1, ))
    assert_size_stride(primals_346, (128, ), (1, ))
    assert_size_stride(primals_347, (128, ), (1, ))
    assert_size_stride(primals_348, (128, ), (1, ))
    assert_size_stride(primals_349, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_350, (32, ), (1, ))
    assert_size_stride(primals_351, (32, ), (1, ))
    assert_size_stride(primals_352, (32, ), (1, ))
    assert_size_stride(primals_353, (32, ), (1, ))
    assert_size_stride(primals_354, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_355, (32, ), (1, ))
    assert_size_stride(primals_356, (32, ), (1, ))
    assert_size_stride(primals_357, (32, ), (1, ))
    assert_size_stride(primals_358, (32, ), (1, ))
    assert_size_stride(primals_359, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_360, (128, ), (1, ))
    assert_size_stride(primals_361, (128, ), (1, ))
    assert_size_stride(primals_362, (128, ), (1, ))
    assert_size_stride(primals_363, (128, ), (1, ))
    assert_size_stride(primals_364, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_365, (32, ), (1, ))
    assert_size_stride(primals_366, (32, ), (1, ))
    assert_size_stride(primals_367, (32, ), (1, ))
    assert_size_stride(primals_368, (32, ), (1, ))
    assert_size_stride(primals_369, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_370, (32, ), (1, ))
    assert_size_stride(primals_371, (32, ), (1, ))
    assert_size_stride(primals_372, (32, ), (1, ))
    assert_size_stride(primals_373, (32, ), (1, ))
    assert_size_stride(primals_374, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_375, (128, ), (1, ))
    assert_size_stride(primals_376, (128, ), (1, ))
    assert_size_stride(primals_377, (128, ), (1, ))
    assert_size_stride(primals_378, (128, ), (1, ))
    assert_size_stride(primals_379, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_380, (32, ), (1, ))
    assert_size_stride(primals_381, (32, ), (1, ))
    assert_size_stride(primals_382, (32, ), (1, ))
    assert_size_stride(primals_383, (32, ), (1, ))
    assert_size_stride(primals_384, (32, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_385, (32, 32, 1, 5), (160, 5, 5, 1))
    assert_size_stride(primals_386, (32, ), (1, ))
    assert_size_stride(primals_387, (32, ), (1, ))
    assert_size_stride(primals_388, (32, ), (1, ))
    assert_size_stride(primals_389, (32, ), (1, ))
    assert_size_stride(primals_390, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_391, (128, ), (1, ))
    assert_size_stride(primals_392, (128, ), (1, ))
    assert_size_stride(primals_393, (128, ), (1, ))
    assert_size_stride(primals_394, (128, ), (1, ))
    assert_size_stride(primals_395, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_396, (32, ), (1, ))
    assert_size_stride(primals_397, (32, ), (1, ))
    assert_size_stride(primals_398, (32, ), (1, ))
    assert_size_stride(primals_399, (32, ), (1, ))
    assert_size_stride(primals_400, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_401, (32, ), (1, ))
    assert_size_stride(primals_402, (32, ), (1, ))
    assert_size_stride(primals_403, (32, ), (1, ))
    assert_size_stride(primals_404, (32, ), (1, ))
    assert_size_stride(primals_405, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_406, (128, ), (1, ))
    assert_size_stride(primals_407, (128, ), (1, ))
    assert_size_stride(primals_408, (128, ), (1, ))
    assert_size_stride(primals_409, (128, ), (1, ))
    assert_size_stride(primals_410, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_411, (32, ), (1, ))
    assert_size_stride(primals_412, (32, ), (1, ))
    assert_size_stride(primals_413, (32, ), (1, ))
    assert_size_stride(primals_414, (32, ), (1, ))
    assert_size_stride(primals_415, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_416, (32, ), (1, ))
    assert_size_stride(primals_417, (32, ), (1, ))
    assert_size_stride(primals_418, (32, ), (1, ))
    assert_size_stride(primals_419, (32, ), (1, ))
    assert_size_stride(primals_420, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_421, (128, ), (1, ))
    assert_size_stride(primals_422, (128, ), (1, ))
    assert_size_stride(primals_423, (128, ), (1, ))
    assert_size_stride(primals_424, (128, ), (1, ))
    assert_size_stride(primals_425, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_426, (32, ), (1, ))
    assert_size_stride(primals_427, (32, ), (1, ))
    assert_size_stride(primals_428, (32, ), (1, ))
    assert_size_stride(primals_429, (32, ), (1, ))
    assert_size_stride(primals_430, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_431, (32, ), (1, ))
    assert_size_stride(primals_432, (32, ), (1, ))
    assert_size_stride(primals_433, (32, ), (1, ))
    assert_size_stride(primals_434, (32, ), (1, ))
    assert_size_stride(primals_435, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_436, (128, ), (1, ))
    assert_size_stride(primals_437, (128, ), (1, ))
    assert_size_stride(primals_438, (128, ), (1, ))
    assert_size_stride(primals_439, (128, ), (1, ))
    assert_size_stride(primals_440, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_441, (32, ), (1, ))
    assert_size_stride(primals_442, (32, ), (1, ))
    assert_size_stride(primals_443, (32, ), (1, ))
    assert_size_stride(primals_444, (32, ), (1, ))
    assert_size_stride(primals_445, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_446, (32, ), (1, ))
    assert_size_stride(primals_447, (32, ), (1, ))
    assert_size_stride(primals_448, (32, ), (1, ))
    assert_size_stride(primals_449, (32, ), (1, ))
    assert_size_stride(primals_450, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_451, (128, ), (1, ))
    assert_size_stride(primals_452, (128, ), (1, ))
    assert_size_stride(primals_453, (128, ), (1, ))
    assert_size_stride(primals_454, (128, ), (1, ))
    assert_size_stride(primals_455, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_456, (32, ), (1, ))
    assert_size_stride(primals_457, (32, ), (1, ))
    assert_size_stride(primals_458, (32, ), (1, ))
    assert_size_stride(primals_459, (32, ), (1, ))
    assert_size_stride(primals_460, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_461, (32, ), (1, ))
    assert_size_stride(primals_462, (32, ), (1, ))
    assert_size_stride(primals_463, (32, ), (1, ))
    assert_size_stride(primals_464, (32, ), (1, ))
    assert_size_stride(primals_465, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_466, (128, ), (1, ))
    assert_size_stride(primals_467, (128, ), (1, ))
    assert_size_stride(primals_468, (128, ), (1, ))
    assert_size_stride(primals_469, (128, ), (1, ))
    assert_size_stride(primals_470, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_471, (32, ), (1, ))
    assert_size_stride(primals_472, (32, ), (1, ))
    assert_size_stride(primals_473, (32, ), (1, ))
    assert_size_stride(primals_474, (32, ), (1, ))
    assert_size_stride(primals_475, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_476, (32, ), (1, ))
    assert_size_stride(primals_477, (32, ), (1, ))
    assert_size_stride(primals_478, (32, ), (1, ))
    assert_size_stride(primals_479, (32, ), (1, ))
    assert_size_stride(primals_480, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_481, (128, ), (1, ))
    assert_size_stride(primals_482, (128, ), (1, ))
    assert_size_stride(primals_483, (128, ), (1, ))
    assert_size_stride(primals_484, (128, ), (1, ))
    assert_size_stride(primals_485, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_486, (32, ), (1, ))
    assert_size_stride(primals_487, (32, ), (1, ))
    assert_size_stride(primals_488, (32, ), (1, ))
    assert_size_stride(primals_489, (32, ), (1, ))
    assert_size_stride(primals_490, (32, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_491, (32, 32, 1, 5), (160, 5, 5, 1))
    assert_size_stride(primals_492, (32, ), (1, ))
    assert_size_stride(primals_493, (32, ), (1, ))
    assert_size_stride(primals_494, (32, ), (1, ))
    assert_size_stride(primals_495, (32, ), (1, ))
    assert_size_stride(primals_496, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_497, (128, ), (1, ))
    assert_size_stride(primals_498, (128, ), (1, ))
    assert_size_stride(primals_499, (128, ), (1, ))
    assert_size_stride(primals_500, (128, ), (1, ))
    assert_size_stride(primals_501, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_502, (32, ), (1, ))
    assert_size_stride(primals_503, (32, ), (1, ))
    assert_size_stride(primals_504, (32, ), (1, ))
    assert_size_stride(primals_505, (32, ), (1, ))
    assert_size_stride(primals_506, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_507, (32, ), (1, ))
    assert_size_stride(primals_508, (32, ), (1, ))
    assert_size_stride(primals_509, (32, ), (1, ))
    assert_size_stride(primals_510, (32, ), (1, ))
    assert_size_stride(primals_511, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_512, (128, ), (1, ))
    assert_size_stride(primals_513, (128, ), (1, ))
    assert_size_stride(primals_514, (128, ), (1, ))
    assert_size_stride(primals_515, (128, ), (1, ))
    assert_size_stride(primals_516, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_517, (64, ), (1, ))
    assert_size_stride(primals_518, (64, ), (1, ))
    assert_size_stride(primals_519, (64, ), (1, ))
    assert_size_stride(primals_520, (64, ), (1, ))
    assert_size_stride(primals_521, (16, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_522, (16, ), (1, ))
    assert_size_stride(primals_523, (16, ), (1, ))
    assert_size_stride(primals_524, (16, ), (1, ))
    assert_size_stride(primals_525, (16, ), (1, ))
    assert_size_stride(primals_526, (16, 16, 2, 2), (64, 4, 2, 1))
    assert_size_stride(primals_527, (16, ), (1, ))
    assert_size_stride(primals_528, (16, ), (1, ))
    assert_size_stride(primals_529, (16, ), (1, ))
    assert_size_stride(primals_530, (16, ), (1, ))
    assert_size_stride(primals_531, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_532, (64, ), (1, ))
    assert_size_stride(primals_533, (64, ), (1, ))
    assert_size_stride(primals_534, (64, ), (1, ))
    assert_size_stride(primals_535, (64, ), (1, ))
    assert_size_stride(primals_536, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_537, (16, ), (1, ))
    assert_size_stride(primals_538, (16, ), (1, ))
    assert_size_stride(primals_539, (16, ), (1, ))
    assert_size_stride(primals_540, (16, ), (1, ))
    assert_size_stride(primals_541, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_542, (16, ), (1, ))
    assert_size_stride(primals_543, (16, ), (1, ))
    assert_size_stride(primals_544, (16, ), (1, ))
    assert_size_stride(primals_545, (16, ), (1, ))
    assert_size_stride(primals_546, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_547, (64, ), (1, ))
    assert_size_stride(primals_548, (64, ), (1, ))
    assert_size_stride(primals_549, (64, ), (1, ))
    assert_size_stride(primals_550, (64, ), (1, ))
    assert_size_stride(primals_551, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_552, (16, ), (1, ))
    assert_size_stride(primals_553, (16, ), (1, ))
    assert_size_stride(primals_554, (16, ), (1, ))
    assert_size_stride(primals_555, (16, ), (1, ))
    assert_size_stride(primals_556, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_557, (16, ), (1, ))
    assert_size_stride(primals_558, (16, ), (1, ))
    assert_size_stride(primals_559, (16, ), (1, ))
    assert_size_stride(primals_560, (16, ), (1, ))
    assert_size_stride(primals_561, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_562, (64, ), (1, ))
    assert_size_stride(primals_563, (64, ), (1, ))
    assert_size_stride(primals_564, (64, ), (1, ))
    assert_size_stride(primals_565, (64, ), (1, ))
    assert_size_stride(primals_566, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_567, (16, ), (1, ))
    assert_size_stride(primals_568, (16, ), (1, ))
    assert_size_stride(primals_569, (16, ), (1, ))
    assert_size_stride(primals_570, (16, ), (1, ))
    assert_size_stride(primals_571, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_572, (16, ), (1, ))
    assert_size_stride(primals_573, (16, ), (1, ))
    assert_size_stride(primals_574, (16, ), (1, ))
    assert_size_stride(primals_575, (16, ), (1, ))
    assert_size_stride(primals_576, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_577, (64, ), (1, ))
    assert_size_stride(primals_578, (64, ), (1, ))
    assert_size_stride(primals_579, (64, ), (1, ))
    assert_size_stride(primals_580, (64, ), (1, ))
    assert_size_stride(primals_581, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_582, (16, ), (1, ))
    assert_size_stride(primals_583, (16, ), (1, ))
    assert_size_stride(primals_584, (16, ), (1, ))
    assert_size_stride(primals_585, (16, ), (1, ))
    assert_size_stride(primals_586, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_587, (16, ), (1, ))
    assert_size_stride(primals_588, (16, ), (1, ))
    assert_size_stride(primals_589, (16, ), (1, ))
    assert_size_stride(primals_590, (16, ), (1, ))
    assert_size_stride(primals_591, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_592, (64, ), (1, ))
    assert_size_stride(primals_593, (64, ), (1, ))
    assert_size_stride(primals_594, (64, ), (1, ))
    assert_size_stride(primals_595, (64, ), (1, ))
    assert_size_stride(primals_596, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_597, (16, ), (1, ))
    assert_size_stride(primals_598, (16, ), (1, ))
    assert_size_stride(primals_599, (16, ), (1, ))
    assert_size_stride(primals_600, (16, ), (1, ))
    assert_size_stride(primals_601, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_602, (16, ), (1, ))
    assert_size_stride(primals_603, (16, ), (1, ))
    assert_size_stride(primals_604, (16, ), (1, ))
    assert_size_stride(primals_605, (16, ), (1, ))
    assert_size_stride(primals_606, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_607, (64, ), (1, ))
    assert_size_stride(primals_608, (64, ), (1, ))
    assert_size_stride(primals_609, (64, ), (1, ))
    assert_size_stride(primals_610, (64, ), (1, ))
    assert_size_stride(primals_611, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_612, (16, ), (1, ))
    assert_size_stride(primals_613, (16, ), (1, ))
    assert_size_stride(primals_614, (16, ), (1, ))
    assert_size_stride(primals_615, (16, ), (1, ))
    assert_size_stride(primals_616, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_617, (16, ), (1, ))
    assert_size_stride(primals_618, (16, ), (1, ))
    assert_size_stride(primals_619, (16, ), (1, ))
    assert_size_stride(primals_620, (16, ), (1, ))
    assert_size_stride(primals_621, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_622, (64, ), (1, ))
    assert_size_stride(primals_623, (64, ), (1, ))
    assert_size_stride(primals_624, (64, ), (1, ))
    assert_size_stride(primals_625, (64, ), (1, ))
    assert_size_stride(primals_626, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_627, (16, ), (1, ))
    assert_size_stride(primals_628, (16, ), (1, ))
    assert_size_stride(primals_629, (16, ), (1, ))
    assert_size_stride(primals_630, (16, ), (1, ))
    assert_size_stride(primals_631, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_632, (16, ), (1, ))
    assert_size_stride(primals_633, (16, ), (1, ))
    assert_size_stride(primals_634, (16, ), (1, ))
    assert_size_stride(primals_635, (16, ), (1, ))
    assert_size_stride(primals_636, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_637, (64, ), (1, ))
    assert_size_stride(primals_638, (64, ), (1, ))
    assert_size_stride(primals_639, (64, ), (1, ))
    assert_size_stride(primals_640, (64, ), (1, ))
    assert_size_stride(primals_641, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_642, (16, ), (1, ))
    assert_size_stride(primals_643, (16, ), (1, ))
    assert_size_stride(primals_644, (16, ), (1, ))
    assert_size_stride(primals_645, (16, ), (1, ))
    assert_size_stride(primals_646, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_647, (16, ), (1, ))
    assert_size_stride(primals_648, (16, ), (1, ))
    assert_size_stride(primals_649, (16, ), (1, ))
    assert_size_stride(primals_650, (16, ), (1, ))
    assert_size_stride(primals_651, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_652, (64, ), (1, ))
    assert_size_stride(primals_653, (64, ), (1, ))
    assert_size_stride(primals_654, (64, ), (1, ))
    assert_size_stride(primals_655, (64, ), (1, ))
    assert_size_stride(primals_656, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_657, (16, ), (1, ))
    assert_size_stride(primals_658, (16, ), (1, ))
    assert_size_stride(primals_659, (16, ), (1, ))
    assert_size_stride(primals_660, (16, ), (1, ))
    assert_size_stride(primals_661, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_662, (16, ), (1, ))
    assert_size_stride(primals_663, (16, ), (1, ))
    assert_size_stride(primals_664, (16, ), (1, ))
    assert_size_stride(primals_665, (16, ), (1, ))
    assert_size_stride(primals_666, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_667, (64, ), (1, ))
    assert_size_stride(primals_668, (64, ), (1, ))
    assert_size_stride(primals_669, (64, ), (1, ))
    assert_size_stride(primals_670, (64, ), (1, ))
    assert_size_stride(primals_671, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_672, (16, ), (1, ))
    assert_size_stride(primals_673, (16, ), (1, ))
    assert_size_stride(primals_674, (16, ), (1, ))
    assert_size_stride(primals_675, (16, ), (1, ))
    assert_size_stride(primals_676, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_677, (16, ), (1, ))
    assert_size_stride(primals_678, (16, ), (1, ))
    assert_size_stride(primals_679, (16, ), (1, ))
    assert_size_stride(primals_680, (16, ), (1, ))
    assert_size_stride(primals_681, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_682, (64, ), (1, ))
    assert_size_stride(primals_683, (64, ), (1, ))
    assert_size_stride(primals_684, (64, ), (1, ))
    assert_size_stride(primals_685, (64, ), (1, ))
    assert_size_stride(primals_686, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_687, (16, ), (1, ))
    assert_size_stride(primals_688, (16, ), (1, ))
    assert_size_stride(primals_689, (16, ), (1, ))
    assert_size_stride(primals_690, (16, ), (1, ))
    assert_size_stride(primals_691, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_692, (4, ), (1, ))
    assert_size_stride(primals_693, (4, ), (1, ))
    assert_size_stride(primals_694, (4, ), (1, ))
    assert_size_stride(primals_695, (4, ), (1, ))
    assert_size_stride(primals_696, (4, 4, 2, 2), (16, 4, 2, 1))
    assert_size_stride(primals_697, (4, ), (1, ))
    assert_size_stride(primals_698, (4, ), (1, ))
    assert_size_stride(primals_699, (4, ), (1, ))
    assert_size_stride(primals_700, (4, ), (1, ))
    assert_size_stride(primals_701, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_702, (16, ), (1, ))
    assert_size_stride(primals_703, (16, ), (1, ))
    assert_size_stride(primals_704, (16, ), (1, ))
    assert_size_stride(primals_705, (16, ), (1, ))
    assert_size_stride(primals_706, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_707, (4, ), (1, ))
    assert_size_stride(primals_708, (4, ), (1, ))
    assert_size_stride(primals_709, (4, ), (1, ))
    assert_size_stride(primals_710, (4, ), (1, ))
    assert_size_stride(primals_711, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_712, (4, ), (1, ))
    assert_size_stride(primals_713, (4, ), (1, ))
    assert_size_stride(primals_714, (4, ), (1, ))
    assert_size_stride(primals_715, (4, ), (1, ))
    assert_size_stride(primals_716, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_717, (16, ), (1, ))
    assert_size_stride(primals_718, (16, ), (1, ))
    assert_size_stride(primals_719, (16, ), (1, ))
    assert_size_stride(primals_720, (16, ), (1, ))
    assert_size_stride(primals_721, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_722, (4, ), (1, ))
    assert_size_stride(primals_723, (4, ), (1, ))
    assert_size_stride(primals_724, (4, ), (1, ))
    assert_size_stride(primals_725, (4, ), (1, ))
    assert_size_stride(primals_726, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_727, (4, ), (1, ))
    assert_size_stride(primals_728, (4, ), (1, ))
    assert_size_stride(primals_729, (4, ), (1, ))
    assert_size_stride(primals_730, (4, ), (1, ))
    assert_size_stride(primals_731, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_732, (16, ), (1, ))
    assert_size_stride(primals_733, (16, ), (1, ))
    assert_size_stride(primals_734, (16, ), (1, ))
    assert_size_stride(primals_735, (16, ), (1, ))
    assert_size_stride(primals_736, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_737, (4, ), (1, ))
    assert_size_stride(primals_738, (4, ), (1, ))
    assert_size_stride(primals_739, (4, ), (1, ))
    assert_size_stride(primals_740, (4, ), (1, ))
    assert_size_stride(primals_741, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_742, (4, ), (1, ))
    assert_size_stride(primals_743, (4, ), (1, ))
    assert_size_stride(primals_744, (4, ), (1, ))
    assert_size_stride(primals_745, (4, ), (1, ))
    assert_size_stride(primals_746, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_747, (16, ), (1, ))
    assert_size_stride(primals_748, (16, ), (1, ))
    assert_size_stride(primals_749, (16, ), (1, ))
    assert_size_stride(primals_750, (16, ), (1, ))
    assert_size_stride(primals_751, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_752, (4, ), (1, ))
    assert_size_stride(primals_753, (4, ), (1, ))
    assert_size_stride(primals_754, (4, ), (1, ))
    assert_size_stride(primals_755, (4, ), (1, ))
    assert_size_stride(primals_756, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_757, (4, ), (1, ))
    assert_size_stride(primals_758, (4, ), (1, ))
    assert_size_stride(primals_759, (4, ), (1, ))
    assert_size_stride(primals_760, (4, ), (1, ))
    assert_size_stride(primals_761, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_762, (16, ), (1, ))
    assert_size_stride(primals_763, (16, ), (1, ))
    assert_size_stride(primals_764, (16, ), (1, ))
    assert_size_stride(primals_765, (16, ), (1, ))
    assert_size_stride(primals_766, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_767, (4, ), (1, ))
    assert_size_stride(primals_768, (4, ), (1, ))
    assert_size_stride(primals_769, (4, ), (1, ))
    assert_size_stride(primals_770, (4, ), (1, ))
    assert_size_stride(primals_771, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_772, (4, ), (1, ))
    assert_size_stride(primals_773, (4, ), (1, ))
    assert_size_stride(primals_774, (4, ), (1, ))
    assert_size_stride(primals_775, (4, ), (1, ))
    assert_size_stride(primals_776, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_777, (16, ), (1, ))
    assert_size_stride(primals_778, (16, ), (1, ))
    assert_size_stride(primals_779, (16, ), (1, ))
    assert_size_stride(primals_780, (16, ), (1, ))
    assert_size_stride(primals_781, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_782, (4, ), (1, ))
    assert_size_stride(primals_783, (4, ), (1, ))
    assert_size_stride(primals_784, (4, ), (1, ))
    assert_size_stride(primals_785, (4, ), (1, ))
    assert_size_stride(primals_786, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_787, (4, ), (1, ))
    assert_size_stride(primals_788, (4, ), (1, ))
    assert_size_stride(primals_789, (4, ), (1, ))
    assert_size_stride(primals_790, (4, ), (1, ))
    assert_size_stride(primals_791, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_792, (16, ), (1, ))
    assert_size_stride(primals_793, (16, ), (1, ))
    assert_size_stride(primals_794, (16, ), (1, ))
    assert_size_stride(primals_795, (16, ), (1, ))
    assert_size_stride(primals_796, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_797, (4, ), (1, ))
    assert_size_stride(primals_798, (4, ), (1, ))
    assert_size_stride(primals_799, (4, ), (1, ))
    assert_size_stride(primals_800, (4, ), (1, ))
    assert_size_stride(primals_801, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_802, (4, ), (1, ))
    assert_size_stride(primals_803, (4, ), (1, ))
    assert_size_stride(primals_804, (4, ), (1, ))
    assert_size_stride(primals_805, (4, ), (1, ))
    assert_size_stride(primals_806, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_807, (16, ), (1, ))
    assert_size_stride(primals_808, (16, ), (1, ))
    assert_size_stride(primals_809, (16, ), (1, ))
    assert_size_stride(primals_810, (16, ), (1, ))
    assert_size_stride(primals_811, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_812, (4, ), (1, ))
    assert_size_stride(primals_813, (4, ), (1, ))
    assert_size_stride(primals_814, (4, ), (1, ))
    assert_size_stride(primals_815, (4, ), (1, ))
    assert_size_stride(primals_816, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_817, (4, ), (1, ))
    assert_size_stride(primals_818, (4, ), (1, ))
    assert_size_stride(primals_819, (4, ), (1, ))
    assert_size_stride(primals_820, (4, ), (1, ))
    assert_size_stride(primals_821, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_822, (16, ), (1, ))
    assert_size_stride(primals_823, (16, ), (1, ))
    assert_size_stride(primals_824, (16, ), (1, ))
    assert_size_stride(primals_825, (16, ), (1, ))
    assert_size_stride(primals_826, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_827, (4, ), (1, ))
    assert_size_stride(primals_828, (4, ), (1, ))
    assert_size_stride(primals_829, (4, ), (1, ))
    assert_size_stride(primals_830, (4, ), (1, ))
    assert_size_stride(primals_831, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_832, (4, ), (1, ))
    assert_size_stride(primals_833, (4, ), (1, ))
    assert_size_stride(primals_834, (4, ), (1, ))
    assert_size_stride(primals_835, (4, ), (1, ))
    assert_size_stride(primals_836, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_837, (16, ), (1, ))
    assert_size_stride(primals_838, (16, ), (1, ))
    assert_size_stride(primals_839, (16, ), (1, ))
    assert_size_stride(primals_840, (16, ), (1, ))
    assert_size_stride(primals_841, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_842, (4, ), (1, ))
    assert_size_stride(primals_843, (4, ), (1, ))
    assert_size_stride(primals_844, (4, ), (1, ))
    assert_size_stride(primals_845, (4, ), (1, ))
    assert_size_stride(primals_846, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_847, (4, ), (1, ))
    assert_size_stride(primals_848, (4, ), (1, ))
    assert_size_stride(primals_849, (4, ), (1, ))
    assert_size_stride(primals_850, (4, ), (1, ))
    assert_size_stride(primals_851, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_852, (16, ), (1, ))
    assert_size_stride(primals_853, (16, ), (1, ))
    assert_size_stride(primals_854, (16, ), (1, ))
    assert_size_stride(primals_855, (16, ), (1, ))
    assert_size_stride(primals_856, (16, 4, 2, 2), (16, 4, 2, 1))
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
        buf2 = empty_strided_cuda((16, 16, 2, 2), (64, 1, 32, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_17, buf2, 256, 4, grid=grid(256, 4), stream=stream0)
        del primals_17
        buf3 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_32, buf3, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_32
        buf4 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_47, buf4, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_47
        buf5 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_62, buf5, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_62
        buf6 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_77, buf6, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_77
        buf7 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_92, buf7, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_92
        buf8 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_107, buf8, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_107
        buf9 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_122, buf9, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_122
        buf10 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_137, buf10, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_137
        buf11 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_152, buf11, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_152
        buf12 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_167, buf12, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_167
        buf13 = empty_strided_cuda((32, 32, 2, 2), (128, 1, 64, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_187, buf13, 1024, 4, grid=grid(1024, 4), stream=stream0)
        del primals_187
        buf14 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_202, buf14, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_202
        buf15 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_217, buf15, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_217
        buf16 = empty_strided_cuda((32, 32, 5, 1), (160, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_232, buf16, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_232
        buf17 = empty_strided_cuda((32, 32, 1, 5), (160, 1, 160, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_233, buf17, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_233
        buf18 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_248, buf18, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_248
        buf19 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_263, buf19, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_263
        buf20 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_278, buf20, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_278
        buf21 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_293, buf21, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_293
        buf22 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_308, buf22, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_308
        buf23 = empty_strided_cuda((32, 32, 5, 1), (160, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_323, buf23, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_323
        buf24 = empty_strided_cuda((32, 32, 1, 5), (160, 1, 160, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_324, buf24, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_324
        buf25 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_339, buf25, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_339
        buf26 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_354, buf26, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_354
        buf27 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_369, buf27, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_369
        buf28 = empty_strided_cuda((32, 32, 5, 1), (160, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_384, buf28, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_384
        buf29 = empty_strided_cuda((32, 32, 1, 5), (160, 1, 160, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_385, buf29, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_385
        buf30 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_400, buf30, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_400
        buf31 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_415, buf31, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_415
        buf32 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_430, buf32, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_430
        buf33 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_445, buf33, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_445
        buf34 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_460, buf34, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_460
        buf35 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_475, buf35, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_475
        buf36 = empty_strided_cuda((32, 32, 5, 1), (160, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_490, buf36, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_490
        buf37 = empty_strided_cuda((32, 32, 1, 5), (160, 1, 160, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_491, buf37, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_491
        buf38 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_506, buf38, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_506
        buf39 = empty_strided_cuda((16, 16, 2, 2), (64, 1, 32, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_526, buf39, 256, 4, grid=grid(256, 4), stream=stream0)
        del primals_526
        buf40 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_541, buf40, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_541
        buf41 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_556, buf41, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_556
        buf42 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_571, buf42, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_571
        buf43 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_586, buf43, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_586
        buf44 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_601, buf44, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_601
        buf45 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_616, buf45, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_616
        buf46 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_631, buf46, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_631
        buf47 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_646, buf47, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_646
        buf48 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_661, buf48, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_661
        buf49 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_676, buf49, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_676
        buf50 = empty_strided_cuda((4, 4, 2, 2), (16, 1, 8, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_696, buf50, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_696
        buf51 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_711, buf51, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_711
        buf52 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_726, buf52, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_726
        buf53 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_741, buf53, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_741
        buf54 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_756, buf54, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_756
        buf55 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_771, buf55, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_771
        buf56 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_786, buf56, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_786
        buf57 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_801, buf57, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_801
        buf58 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_816, buf58, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_816
        buf59 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_831, buf59, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_831
        buf60 = empty_strided_cuda((4, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_846, buf60, 16, 9, grid=grid(16, 9), stream=stream0)
        del primals_846
        buf61 = empty_strided_cuda((16, 4, 2, 2), (16, 1, 8, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_856, buf61, 64, 4, grid=grid(64, 4), stream=stream0)
        del primals_856
        # Topologically Sorted Source Nodes: [x_conv], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 13, 32, 32), (13312, 1, 416, 13))
        buf63 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf64 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_rrelu_with_noise_functional_10.run(buf65, buf62, buf1, primals_3, primals_4, primals_5, primals_6, buf63, 65536, grid=grid(65536), stream=stream0)
        del buf62
        buf66 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf67 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.int64)
        # Topologically Sorted Source Nodes: [max_pool2d_1], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_11.run(buf65, buf66, buf67, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf66, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 64, 16, 16), (16384, 1, 1024, 64))
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf65, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf70 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_12.run(buf71, buf69, primals_13, primals_14, primals_15, primals_16, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, buf2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf73 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf74 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf74, buf72, primals_18, primals_19, primals_20, primals_21, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf76 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_10, add, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_14.run(buf77, buf75, primals_23, primals_24, primals_25, primals_26, buf68, primals_8, primals_9, primals_10, primals_11, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf79 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf80 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf80, buf78, primals_28, primals_29, primals_30, primals_31, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf82 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf83, buf81, primals_33, primals_34, primals_35, primals_36, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf85 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf86 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf87 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [input_19, add_1, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf87, buf84, primals_38, primals_39, primals_40, primals_41, buf77, buf86, 65536, grid=grid(65536), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf89 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf90 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf90, buf88, primals_43, primals_44, primals_45, primals_46, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf92 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf93, buf91, primals_48, primals_49, primals_50, primals_51, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf95 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf96 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf97 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [input_28, add_2, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf97, buf94, primals_53, primals_54, primals_55, primals_56, buf87, buf96, 65536, grid=grid(65536), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf99 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf100, buf98, primals_58, primals_59, primals_60, primals_61, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf102 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf103, buf101, primals_63, primals_64, primals_65, primals_66, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf105 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf106 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf107 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [input_37, add_3, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf107, buf104, primals_68, primals_69, primals_70, primals_71, buf97, buf106, 65536, grid=grid(65536), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf109 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [input_40, input_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf110, buf108, primals_73, primals_74, primals_75, primals_76, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf112 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf113, buf111, primals_78, primals_79, primals_80, primals_81, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf115 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf116 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf117 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [input_46, add_4, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf117, buf114, primals_83, primals_84, primals_85, primals_86, buf107, buf116, 65536, grid=grid(65536), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf119 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf120 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [input_49, input_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf120, buf118, primals_88, primals_89, primals_90, primals_91, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf122 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf123, buf121, primals_93, primals_94, primals_95, primals_96, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf125 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf126 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf127 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [input_55, add_5, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf127, buf124, primals_98, primals_99, primals_100, primals_101, buf117, buf126, 65536, grid=grid(65536), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf129 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [input_58, input_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf130, buf128, primals_103, primals_104, primals_105, primals_106, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf132 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf133 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [input_61, input_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf133, buf131, primals_108, primals_109, primals_110, primals_111, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf135 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf136 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf137 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [input_64, add_6, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf137, buf134, primals_113, primals_114, primals_115, primals_116, buf127, buf136, 65536, grid=grid(65536), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf139 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf140 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [input_67, input_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf140, buf138, primals_118, primals_119, primals_120, primals_121, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf142 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf143 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [input_70, input_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf143, buf141, primals_123, primals_124, primals_125, primals_126, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf145 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf146 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf147 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [input_73, add_7, out_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf147, buf144, primals_128, primals_129, primals_130, primals_131, buf137, buf146, 65536, grid=grid(65536), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf149 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [input_76, input_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf150, buf148, primals_133, primals_134, primals_135, primals_136, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf152 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf153 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [input_79, input_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf153, buf151, primals_138, primals_139, primals_140, primals_141, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf155 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf156 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf157 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [input_82, add_8, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf157, buf154, primals_143, primals_144, primals_145, primals_146, buf147, buf156, 65536, grid=grid(65536), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf159 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf160 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [input_85, input_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf160, buf158, primals_148, primals_149, primals_150, primals_151, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf162 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf163 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [input_88, input_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf163, buf161, primals_153, primals_154, primals_155, primals_156, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf165 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf166 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf167 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [input_91, add_9, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf167, buf164, primals_158, primals_159, primals_160, primals_161, buf157, buf166, 65536, grid=grid(65536), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf169 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf170 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [input_94, input_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf170, buf168, primals_163, primals_164, primals_165, primals_166, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf172 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf173 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [input_97, input_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf173, buf171, primals_168, primals_169, primals_170, primals_171, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf175 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf176 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf177 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [input_100, add_10, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf177, buf174, primals_173, primals_174, primals_175, primals_176, buf167, buf176, 65536, grid=grid(65536), stream=stream0)
        del primals_176
        buf178 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        buf179 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.int64)
        # Topologically Sorted Source Nodes: [max_pool2d_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_16.run(buf177, buf178, buf179, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf178, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 128, 8, 8), (8192, 1, 1024, 128))
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf177, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf182 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.float32)
        buf183 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [input_105, input_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_17.run(buf183, buf181, primals_183, primals_184, primals_185, primals_186, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_107], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, buf13, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf185 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf186 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [input_108, input_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf186, buf184, primals_188, primals_189, primals_190, primals_191, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf188 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [input_103, input_111, add_11, out_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_19.run(buf189, buf187, primals_193, primals_194, primals_195, primals_196, buf180, primals_178, primals_179, primals_180, primals_181, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_113], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf191 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [input_114, input_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf192, buf190, primals_198, primals_199, primals_200, primals_201, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf194 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf195 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [input_117, input_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf195, buf193, primals_203, primals_204, primals_205, primals_206, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf197 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf198 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf199 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [input_120, add_12, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf199, buf196, primals_208, primals_209, primals_210, primals_211, buf189, buf198, 32768, grid=grid(32768), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf201 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf202 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [input_123, input_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf202, buf200, primals_213, primals_214, primals_215, primals_216, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, buf15, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf204 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [input_126, input_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf205, buf203, primals_218, primals_219, primals_220, primals_221, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_128], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf207 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf208 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf209 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [input_129, add_13, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf209, buf206, primals_223, primals_224, primals_225, primals_226, buf199, buf208, 32768, grid=grid(32768), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf211 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf212 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [input_132, input_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf212, buf210, primals_228, primals_229, primals_230, primals_231, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, buf16, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 32, 8, 8), (2048, 1, 256, 32))
        # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, buf17, stride=(1, 1), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf215 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf216 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [input_136, input_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf216, buf214, primals_234, primals_235, primals_236, primals_237, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_138], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf218 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf219 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf220 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [input_139, add_14, out_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf220, buf217, primals_239, primals_240, primals_241, primals_242, buf209, buf219, 32768, grid=grid(32768), stream=stream0)
        del primals_242
        # Topologically Sorted Source Nodes: [input_141], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf222 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf223 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [input_142, input_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf223, buf221, primals_244, primals_245, primals_246, primals_247, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_144], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, buf18, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf225 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf226 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [input_145, input_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf226, buf224, primals_249, primals_250, primals_251, primals_252, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf228 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf229 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf230 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [input_148, add_15, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf230, buf227, primals_254, primals_255, primals_256, primals_257, buf220, buf229, 32768, grid=grid(32768), stream=stream0)
        del primals_257
        # Topologically Sorted Source Nodes: [input_150], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf232 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf233 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [input_151, input_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf233, buf231, primals_259, primals_260, primals_261, primals_262, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_153], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf235 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf236 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [input_154, input_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf236, buf234, primals_264, primals_265, primals_266, primals_267, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_268, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf238 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf239 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf240 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [input_157, add_16, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf240, buf237, primals_269, primals_270, primals_271, primals_272, buf230, buf239, 32768, grid=grid(32768), stream=stream0)
        del primals_272
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf242 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf243 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [input_160, input_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf243, buf241, primals_274, primals_275, primals_276, primals_277, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf245 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf246 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [input_163, input_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf246, buf244, primals_279, primals_280, primals_281, primals_282, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_165], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_283, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf248 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf249 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf250 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [input_166, add_17, out_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf250, buf247, primals_284, primals_285, primals_286, primals_287, buf240, buf249, 32768, grid=grid(32768), stream=stream0)
        del primals_287
        # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_288, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf252 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [input_169, input_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf253, buf251, primals_289, primals_290, primals_291, primals_292, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf255 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf256 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [input_172, input_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf256, buf254, primals_294, primals_295, primals_296, primals_297, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_298, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf258 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf259 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf260 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [input_175, add_18, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf260, buf257, primals_299, primals_300, primals_301, primals_302, buf250, buf259, 32768, grid=grid(32768), stream=stream0)
        del primals_302
        # Topologically Sorted Source Nodes: [input_177], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf262 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf263 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [input_178, input_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf263, buf261, primals_304, primals_305, primals_306, primals_307, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, buf22, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf265 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf266 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [input_181, input_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf266, buf264, primals_309, primals_310, primals_311, primals_312, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_183], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_313, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf268 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf269 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf270 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [input_184, add_19, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf270, buf267, primals_314, primals_315, primals_316, primals_317, buf260, buf269, 32768, grid=grid(32768), stream=stream0)
        del primals_317
        # Topologically Sorted Source Nodes: [input_186], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_318, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf272 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf273 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [input_187, input_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf273, buf271, primals_319, primals_320, primals_321, primals_322, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, buf23, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 32, 8, 8), (2048, 1, 256, 32))
        # Topologically Sorted Source Nodes: [input_190], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, buf24, stride=(1, 1), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf276 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf277 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [input_191, input_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf277, buf275, primals_325, primals_326, primals_327, primals_328, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_193], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_329, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf279 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf280 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf281 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [input_194, add_20, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf281, buf278, primals_330, primals_331, primals_332, primals_333, buf270, buf280, 32768, grid=grid(32768), stream=stream0)
        del primals_333
        # Topologically Sorted Source Nodes: [input_196], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_334, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf283 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf284 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [input_197, input_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf284, buf282, primals_335, primals_336, primals_337, primals_338, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_199], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, buf25, stride=(1, 1), padding=(16, 16), dilation=(16, 16), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf286 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf287 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [input_200, input_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf287, buf285, primals_340, primals_341, primals_342, primals_343, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_202], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_344, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf289 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf290 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf291 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [input_203, add_21, out_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf291, buf288, primals_345, primals_346, primals_347, primals_348, buf281, buf290, 32768, grid=grid(32768), stream=stream0)
        del primals_348
        # Topologically Sorted Source Nodes: [input_205], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_349, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf293 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf294 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [input_206, input_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf294, buf292, primals_350, primals_351, primals_352, primals_353, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_208], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf296 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf297 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [input_209, input_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf297, buf295, primals_355, primals_356, primals_357, primals_358, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_211], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_359, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf299 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf300 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf301 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [input_212, add_22, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf301, buf298, primals_360, primals_361, primals_362, primals_363, buf291, buf300, 32768, grid=grid(32768), stream=stream0)
        del primals_363
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_364, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf303 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf304 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [input_215, input_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf304, buf302, primals_365, primals_366, primals_367, primals_368, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_217], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, buf27, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf306 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf307 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [input_218, input_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf307, buf305, primals_370, primals_371, primals_372, primals_373, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_374, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf309 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf310 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf311 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [input_221, add_23, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf311, buf308, primals_375, primals_376, primals_377, primals_378, buf301, buf310, 32768, grid=grid(32768), stream=stream0)
        del primals_378
        # Topologically Sorted Source Nodes: [input_223], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_379, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf313 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf314 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [input_224, input_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf314, buf312, primals_380, primals_381, primals_382, primals_383, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_226], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, buf28, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (4, 32, 8, 8), (2048, 1, 256, 32))
        # Topologically Sorted Source Nodes: [input_227], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, buf29, stride=(1, 1), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf317 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf318 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [input_228, input_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf318, buf316, primals_386, primals_387, primals_388, primals_389, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_230], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_390, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf320 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf321 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf322 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [input_231, add_24, out_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf322, buf319, primals_391, primals_392, primals_393, primals_394, buf311, buf321, 32768, grid=grid(32768), stream=stream0)
        del primals_394
        # Topologically Sorted Source Nodes: [input_233], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_395, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf324 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf325 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [input_234, input_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf325, buf323, primals_396, primals_397, primals_398, primals_399, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_236], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, buf30, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf327 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf328 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [input_237, input_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf328, buf326, primals_401, primals_402, primals_403, primals_404, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_239], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_405, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf330 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf331 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf332 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [input_240, add_25, out_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf332, buf329, primals_406, primals_407, primals_408, primals_409, buf322, buf331, 32768, grid=grid(32768), stream=stream0)
        del primals_409
        # Topologically Sorted Source Nodes: [input_242], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_410, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf334 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf335 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [input_243, input_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf335, buf333, primals_411, primals_412, primals_413, primals_414, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_245], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf337 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf338 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [input_246, input_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf338, buf336, primals_416, primals_417, primals_418, primals_419, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_248], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_420, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf340 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf341 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf342 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [input_249, add_26, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf342, buf339, primals_421, primals_422, primals_423, primals_424, buf332, buf341, 32768, grid=grid(32768), stream=stream0)
        del primals_424
        # Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, primals_425, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf344 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf345 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [input_252, input_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf345, buf343, primals_426, primals_427, primals_428, primals_429, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_254], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf347 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf348 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [input_255, input_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf348, buf346, primals_431, primals_432, primals_433, primals_434, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_257], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, primals_435, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf350 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf351 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf352 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [input_258, add_27, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf352, buf349, primals_436, primals_437, primals_438, primals_439, buf342, buf351, 32768, grid=grid(32768), stream=stream0)
        del primals_439
        # Topologically Sorted Source Nodes: [input_260], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_440, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf354 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf355 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [input_261, input_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf355, buf353, primals_441, primals_442, primals_443, primals_444, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_263], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf357 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf358 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [input_264, input_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf358, buf356, primals_446, primals_447, primals_448, primals_449, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_266], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_450, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf360 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf361 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf362 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [input_267, add_28, out_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf362, buf359, primals_451, primals_452, primals_453, primals_454, buf352, buf361, 32768, grid=grid(32768), stream=stream0)
        del primals_454
        # Topologically Sorted Source Nodes: [input_269], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, primals_455, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf364 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf365 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [input_270, input_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf365, buf363, primals_456, primals_457, primals_458, primals_459, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_272], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf367 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf368 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [input_273, input_274], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf368, buf366, primals_461, primals_462, primals_463, primals_464, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_275], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, primals_465, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf370 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf371 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf372 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [input_276, add_29, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf372, buf369, primals_466, primals_467, primals_468, primals_469, buf362, buf371, 32768, grid=grid(32768), stream=stream0)
        del primals_469
        # Topologically Sorted Source Nodes: [input_278], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, primals_470, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf374 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf375 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [input_279, input_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf375, buf373, primals_471, primals_472, primals_473, primals_474, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_281], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, buf35, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf377 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf378 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [input_282, input_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf378, buf376, primals_476, primals_477, primals_478, primals_479, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_284], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_480, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf380 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf381 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf382 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [input_285, add_30, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf382, buf379, primals_481, primals_482, primals_483, primals_484, buf372, buf381, 32768, grid=grid(32768), stream=stream0)
        del primals_484
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_485, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf384 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf385 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [input_288, input_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf385, buf383, primals_486, primals_487, primals_488, primals_489, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_290], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, buf36, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (4, 32, 8, 8), (2048, 1, 256, 32))
        # Topologically Sorted Source Nodes: [input_291], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf386, buf37, stride=(1, 1), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf388 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf389 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [input_292, input_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf389, buf387, primals_492, primals_493, primals_494, primals_495, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_294], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf389, primals_496, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf391 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf392 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf393 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [input_295, add_31, out_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf393, buf390, primals_497, primals_498, primals_499, primals_500, buf382, buf392, 32768, grid=grid(32768), stream=stream0)
        del primals_500
        # Topologically Sorted Source Nodes: [input_297], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, primals_501, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf395 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf396 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [input_298, input_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf396, buf394, primals_502, primals_503, primals_504, primals_505, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_300], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, buf38, stride=(1, 1), padding=(16, 16), dilation=(16, 16), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf398 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf399 = buf398; del buf398  # reuse
        # Topologically Sorted Source Nodes: [input_301, input_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_18.run(buf399, buf397, primals_507, primals_508, primals_509, primals_510, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_303], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf399, primals_511, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf401 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf402 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        buf403 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [input_304, add_32, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_20.run(buf403, buf400, primals_512, primals_513, primals_514, primals_515, buf393, buf402, 32768, grid=grid(32768), stream=stream0)
        del primals_515
        # Topologically Sorted Source Nodes: [input_306], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_516, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf405 = empty_strided_cuda((65536, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_up], Original ATen: [aten.max_unpool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_21.run(buf405, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [out_up], Original ATen: [aten.max_unpool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_22.run(buf179, buf404, primals_517, primals_518, primals_519, primals_520, buf405, 16384, grid=grid(16384), stream=stream0)
        del primals_520
        # Topologically Sorted Source Nodes: [input_308], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf403, primals_521, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (4, 16, 8, 8), (1024, 1, 128, 16))
        buf408 = empty_strided_cuda((4, 16, 8, 8), (1024, 1, 128, 16), torch.float32)
        buf409 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [input_309, input_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_23.run(buf409, buf407, primals_522, primals_523, primals_524, primals_525, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_311], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, buf39, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf411 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf412 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [input_312, input_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf412, buf410, primals_527, primals_528, primals_529, primals_530, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_314], Original ATen: [aten.convolution]
        buf413 = extern_kernels.convolution(buf412, primals_531, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf413, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf415 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf416 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_315, add_33, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_24.run(buf405, buf413, primals_532, primals_533, primals_534, primals_535, buf415, buf416, 256, 256, grid=grid(256, 256), stream=stream0)
        del primals_535
        # Topologically Sorted Source Nodes: [input_317], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf416, primals_536, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf418 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf419 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [input_318, input_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf419, buf417, primals_537, primals_538, primals_539, primals_540, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.convolution]
        buf420 = extern_kernels.convolution(buf419, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf421 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf422 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [input_321, input_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf422, buf420, primals_542, primals_543, primals_544, primals_545, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_323], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf422, primals_546, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf423, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf424 = reinterpret_tensor(buf405, (4, 64, 16, 16), (16384, 1, 1024, 64), 0); del buf405  # reuse
        buf425 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf426 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [input_324, add_34, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf426, buf423, primals_547, primals_548, primals_549, primals_550, buf416, buf425, 65536, grid=grid(65536), stream=stream0)
        del primals_550
        # Topologically Sorted Source Nodes: [input_326], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_551, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf428 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf429 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [input_327, input_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf429, buf427, primals_552, primals_553, primals_554, primals_555, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_329], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf431 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf432 = buf431; del buf431  # reuse
        # Topologically Sorted Source Nodes: [input_330, input_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf432, buf430, primals_557, primals_558, primals_559, primals_560, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_332], Original ATen: [aten.convolution]
        buf433 = extern_kernels.convolution(buf432, primals_561, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf434 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf435 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf436 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [input_333, add_35, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf436, buf433, primals_562, primals_563, primals_564, primals_565, buf426, buf435, 65536, grid=grid(65536), stream=stream0)
        del primals_565
        # Topologically Sorted Source Nodes: [input_335], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, primals_566, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf438 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf439 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [input_336, input_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf439, buf437, primals_567, primals_568, primals_569, primals_570, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_338], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf439, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf441 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf442 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [input_339, input_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf442, buf440, primals_572, primals_573, primals_574, primals_575, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_341], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_576, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf444 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf445 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf446 = buf444; del buf444  # reuse
        # Topologically Sorted Source Nodes: [input_342, add_36, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf446, buf443, primals_577, primals_578, primals_579, primals_580, buf436, buf445, 65536, grid=grid(65536), stream=stream0)
        del primals_580
        # Topologically Sorted Source Nodes: [input_344], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, primals_581, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf448 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf449 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [input_345, input_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf449, buf447, primals_582, primals_583, primals_584, primals_585, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_347], Original ATen: [aten.convolution]
        buf450 = extern_kernels.convolution(buf449, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf451 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf452 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [input_348, input_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf452, buf450, primals_587, primals_588, primals_589, primals_590, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_350], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_591, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf454 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf455 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf456 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [input_351, add_37, out_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf456, buf453, primals_592, primals_593, primals_594, primals_595, buf446, buf455, 65536, grid=grid(65536), stream=stream0)
        del primals_595
        # Topologically Sorted Source Nodes: [input_353], Original ATen: [aten.convolution]
        buf457 = extern_kernels.convolution(buf456, primals_596, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf457, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf458 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf459 = buf458; del buf458  # reuse
        # Topologically Sorted Source Nodes: [input_354, input_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf459, buf457, primals_597, primals_598, primals_599, primals_600, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_356], Original ATen: [aten.convolution]
        buf460 = extern_kernels.convolution(buf459, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf460, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf461 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf462 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [input_357, input_358], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf462, buf460, primals_602, primals_603, primals_604, primals_605, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_359], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf462, primals_606, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf464 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf465 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf466 = buf464; del buf464  # reuse
        # Topologically Sorted Source Nodes: [input_360, add_38, out_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf466, buf463, primals_607, primals_608, primals_609, primals_610, buf456, buf465, 65536, grid=grid(65536), stream=stream0)
        del primals_610
        # Topologically Sorted Source Nodes: [input_362], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, primals_611, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf468 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf469 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [input_363, input_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf469, buf467, primals_612, primals_613, primals_614, primals_615, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_365], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf471 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf472 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [input_366, input_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf472, buf470, primals_617, primals_618, primals_619, primals_620, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_368], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, primals_621, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf474 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf475 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf476 = buf474; del buf474  # reuse
        # Topologically Sorted Source Nodes: [input_369, add_39, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf476, buf473, primals_622, primals_623, primals_624, primals_625, buf466, buf475, 65536, grid=grid(65536), stream=stream0)
        del primals_625
        # Topologically Sorted Source Nodes: [input_371], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, primals_626, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf478 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf479 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [input_372, input_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf479, buf477, primals_627, primals_628, primals_629, primals_630, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_374], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf479, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf481 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf482 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [input_375, input_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf482, buf480, primals_632, primals_633, primals_634, primals_635, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_377], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_636, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf484 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf485 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf486 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [input_378, add_40, out_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf486, buf483, primals_637, primals_638, primals_639, primals_640, buf476, buf485, 65536, grid=grid(65536), stream=stream0)
        del primals_640
        # Topologically Sorted Source Nodes: [input_380], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf486, primals_641, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf488 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf489 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [input_381, input_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf489, buf487, primals_642, primals_643, primals_644, primals_645, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_383], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf491 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf492 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [input_384, input_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf492, buf490, primals_647, primals_648, primals_649, primals_650, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_386], Original ATen: [aten.convolution]
        buf493 = extern_kernels.convolution(buf492, primals_651, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf493, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf494 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf495 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf496 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [input_387, add_41, out_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf496, buf493, primals_652, primals_653, primals_654, primals_655, buf486, buf495, 65536, grid=grid(65536), stream=stream0)
        del primals_655
        # Topologically Sorted Source Nodes: [input_389], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, primals_656, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf498 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf499 = buf498; del buf498  # reuse
        # Topologically Sorted Source Nodes: [input_390, input_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf499, buf497, primals_657, primals_658, primals_659, primals_660, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_392], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf499, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf501 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf502 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [input_393, input_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf502, buf500, primals_662, primals_663, primals_664, primals_665, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_395], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf502, primals_666, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf504 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf505 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf506 = buf504; del buf504  # reuse
        # Topologically Sorted Source Nodes: [input_396, add_42, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf506, buf503, primals_667, primals_668, primals_669, primals_670, buf496, buf505, 65536, grid=grid(65536), stream=stream0)
        del primals_670
        # Topologically Sorted Source Nodes: [input_398], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf506, primals_671, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf508 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf509 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [input_399, input_400], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf509, buf507, primals_672, primals_673, primals_674, primals_675, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_401], Original ATen: [aten.convolution]
        buf510 = extern_kernels.convolution(buf509, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf510, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf511 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        buf512 = buf511; del buf511  # reuse
        # Topologically Sorted Source Nodes: [input_402, input_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_13.run(buf512, buf510, primals_677, primals_678, primals_679, primals_680, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_404], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf512, primals_681, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf514 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf515 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        buf516 = buf514; del buf514  # reuse
        # Topologically Sorted Source Nodes: [input_405, add_43, out_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_15.run(buf516, buf513, primals_682, primals_683, primals_684, primals_685, buf506, buf515, 65536, grid=grid(65536), stream=stream0)
        del primals_685
        # Topologically Sorted Source Nodes: [input_407], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, primals_686, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf518 = empty_strided_cuda((65536, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_up], Original ATen: [aten.max_unpool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_21.run(buf518, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [out_up, out_up_1], Original ATen: [aten.max_unpool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_25.run(buf67, buf517, primals_687, primals_688, primals_689, primals_690, buf518, 16384, grid=grid(16384), stream=stream0)
        del primals_690
        # Topologically Sorted Source Nodes: [input_409], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf516, primals_691, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (4, 4, 16, 16), (1024, 1, 64, 4))
        buf521 = empty_strided_cuda((4, 4, 16, 16), (1024, 1, 64, 4), torch.float32)
        buf522 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [input_410, input_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_26.run(buf522, buf520, primals_692, primals_693, primals_694, primals_695, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_412], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, buf50, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf524 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf525 = buf524; del buf524  # reuse
        # Topologically Sorted Source Nodes: [input_413, input_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf525, buf523, primals_697, primals_698, primals_699, primals_700, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_415], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, primals_701, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf528 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf529 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_416, add_44, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_28.run(buf518, buf526, primals_702, primals_703, primals_704, primals_705, buf528, buf529, 64, 1024, grid=grid(64, 1024), stream=stream0)
        del primals_705
        # Topologically Sorted Source Nodes: [input_418], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(buf529, primals_706, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf531 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf532 = buf531; del buf531  # reuse
        # Topologically Sorted Source Nodes: [input_419, input_420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf532, buf530, primals_707, primals_708, primals_709, primals_710, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_421], Original ATen: [aten.convolution]
        buf533 = extern_kernels.convolution(buf532, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf533, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf534 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf535 = buf534; del buf534  # reuse
        # Topologically Sorted Source Nodes: [input_422, input_423], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf535, buf533, primals_712, primals_713, primals_714, primals_715, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_424], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_716, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf537 = reinterpret_tensor(buf518, (4, 16, 32, 32), (16384, 1, 512, 16), 0); del buf518  # reuse
        buf538 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf539 = buf537; del buf537  # reuse
        # Topologically Sorted Source Nodes: [input_425, add_45, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf539, buf536, primals_717, primals_718, primals_719, primals_720, buf529, buf538, 65536, grid=grid(65536), stream=stream0)
        del primals_720
        # Topologically Sorted Source Nodes: [input_427], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, primals_721, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf541 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf542 = buf541; del buf541  # reuse
        # Topologically Sorted Source Nodes: [input_428, input_429], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf542, buf540, primals_722, primals_723, primals_724, primals_725, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_430], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf542, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf544 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf545 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [input_431, input_432], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf545, buf543, primals_727, primals_728, primals_729, primals_730, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_433], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf545, primals_731, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf547 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf548 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf549 = buf547; del buf547  # reuse
        # Topologically Sorted Source Nodes: [input_434, add_46, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf549, buf546, primals_732, primals_733, primals_734, primals_735, buf539, buf548, 65536, grid=grid(65536), stream=stream0)
        del primals_735
        # Topologically Sorted Source Nodes: [input_436], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(buf549, primals_736, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf551 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf552 = buf551; del buf551  # reuse
        # Topologically Sorted Source Nodes: [input_437, input_438], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf552, buf550, primals_737, primals_738, primals_739, primals_740, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_439], Original ATen: [aten.convolution]
        buf553 = extern_kernels.convolution(buf552, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf553, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf554 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf555 = buf554; del buf554  # reuse
        # Topologically Sorted Source Nodes: [input_440, input_441], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf555, buf553, primals_742, primals_743, primals_744, primals_745, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_442], Original ATen: [aten.convolution]
        buf556 = extern_kernels.convolution(buf555, primals_746, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf557 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf558 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf559 = buf557; del buf557  # reuse
        # Topologically Sorted Source Nodes: [input_443, add_47, out_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf559, buf556, primals_747, primals_748, primals_749, primals_750, buf549, buf558, 65536, grid=grid(65536), stream=stream0)
        del primals_750
        # Topologically Sorted Source Nodes: [input_445], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, primals_751, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf561 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf562 = buf561; del buf561  # reuse
        # Topologically Sorted Source Nodes: [input_446, input_447], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf562, buf560, primals_752, primals_753, primals_754, primals_755, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_448], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf564 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf565 = buf564; del buf564  # reuse
        # Topologically Sorted Source Nodes: [input_449, input_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf565, buf563, primals_757, primals_758, primals_759, primals_760, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_451], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(buf565, primals_761, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf567 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf568 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf569 = buf567; del buf567  # reuse
        # Topologically Sorted Source Nodes: [input_452, add_48, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf569, buf566, primals_762, primals_763, primals_764, primals_765, buf559, buf568, 65536, grid=grid(65536), stream=stream0)
        del primals_765
        # Topologically Sorted Source Nodes: [input_454], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, primals_766, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf571 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf572 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [input_455, input_456], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf572, buf570, primals_767, primals_768, primals_769, primals_770, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_457], Original ATen: [aten.convolution]
        buf573 = extern_kernels.convolution(buf572, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf573, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf574 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf575 = buf574; del buf574  # reuse
        # Topologically Sorted Source Nodes: [input_458, input_459], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf575, buf573, primals_772, primals_773, primals_774, primals_775, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_460], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_776, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf577 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf578 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf579 = buf577; del buf577  # reuse
        # Topologically Sorted Source Nodes: [input_461, add_49, out_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf579, buf576, primals_777, primals_778, primals_779, primals_780, buf569, buf578, 65536, grid=grid(65536), stream=stream0)
        del primals_780
        # Topologically Sorted Source Nodes: [input_463], Original ATen: [aten.convolution]
        buf580 = extern_kernels.convolution(buf579, primals_781, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf580, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf581 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf582 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [input_464, input_465], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf582, buf580, primals_782, primals_783, primals_784, primals_785, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_466], Original ATen: [aten.convolution]
        buf583 = extern_kernels.convolution(buf582, buf56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf583, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf584 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf585 = buf584; del buf584  # reuse
        # Topologically Sorted Source Nodes: [input_467, input_468], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf585, buf583, primals_787, primals_788, primals_789, primals_790, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_469], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf585, primals_791, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf587 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf588 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf589 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [input_470, add_50, out_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf589, buf586, primals_792, primals_793, primals_794, primals_795, buf579, buf588, 65536, grid=grid(65536), stream=stream0)
        del primals_795
        # Topologically Sorted Source Nodes: [input_472], Original ATen: [aten.convolution]
        buf590 = extern_kernels.convolution(buf589, primals_796, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf590, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf591 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf592 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [input_473, input_474], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf592, buf590, primals_797, primals_798, primals_799, primals_800, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_475], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf592, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf594 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf595 = buf594; del buf594  # reuse
        # Topologically Sorted Source Nodes: [input_476, input_477], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf595, buf593, primals_802, primals_803, primals_804, primals_805, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_478], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_806, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf597 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf598 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf599 = buf597; del buf597  # reuse
        # Topologically Sorted Source Nodes: [input_479, add_51, out_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf599, buf596, primals_807, primals_808, primals_809, primals_810, buf589, buf598, 65536, grid=grid(65536), stream=stream0)
        del primals_810
        # Topologically Sorted Source Nodes: [input_481], Original ATen: [aten.convolution]
        buf600 = extern_kernels.convolution(buf599, primals_811, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf601 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf602 = buf601; del buf601  # reuse
        # Topologically Sorted Source Nodes: [input_482, input_483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf602, buf600, primals_812, primals_813, primals_814, primals_815, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_484], Original ATen: [aten.convolution]
        buf603 = extern_kernels.convolution(buf602, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf603, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf604 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf605 = buf604; del buf604  # reuse
        # Topologically Sorted Source Nodes: [input_485, input_486], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf605, buf603, primals_817, primals_818, primals_819, primals_820, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_487], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, primals_821, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf607 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf608 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf609 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [input_488, add_52, out_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf609, buf606, primals_822, primals_823, primals_824, primals_825, buf599, buf608, 65536, grid=grid(65536), stream=stream0)
        del primals_825
        # Topologically Sorted Source Nodes: [input_490], Original ATen: [aten.convolution]
        buf610 = extern_kernels.convolution(buf609, primals_826, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf610, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf611 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf612 = buf611; del buf611  # reuse
        # Topologically Sorted Source Nodes: [input_491, input_492], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf612, buf610, primals_827, primals_828, primals_829, primals_830, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_493], Original ATen: [aten.convolution]
        buf613 = extern_kernels.convolution(buf612, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf613, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf614 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf615 = buf614; del buf614  # reuse
        # Topologically Sorted Source Nodes: [input_494, input_495], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf615, buf613, primals_832, primals_833, primals_834, primals_835, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_496], Original ATen: [aten.convolution]
        buf616 = extern_kernels.convolution(buf615, primals_836, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf616, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf617 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf618 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf619 = buf617; del buf617  # reuse
        # Topologically Sorted Source Nodes: [input_497, add_53, out_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf619, buf616, primals_837, primals_838, primals_839, primals_840, buf609, buf618, 65536, grid=grid(65536), stream=stream0)
        del primals_840
        # Topologically Sorted Source Nodes: [input_499], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_841, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf621 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf622 = buf621; del buf621  # reuse
        # Topologically Sorted Source Nodes: [input_500, input_501], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf622, buf620, primals_842, primals_843, primals_844, primals_845, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_502], Original ATen: [aten.convolution]
        buf623 = extern_kernels.convolution(buf622, buf60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf623, (4, 4, 32, 32), (4096, 1, 128, 4))
        buf624 = empty_strided_cuda((4, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        buf625 = buf624; del buf624  # reuse
        # Topologically Sorted Source Nodes: [input_503, input_504], Original ATen: [aten._native_batch_norm_legit_no_training, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_rrelu_with_noise_functional_27.run(buf625, buf623, primals_847, primals_848, primals_849, primals_850, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_505], Original ATen: [aten.convolution]
        buf626 = extern_kernels.convolution(buf625, primals_851, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf626, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf627 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf628 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.bool)
        buf629 = buf627; del buf627  # reuse
        # Topologically Sorted Source Nodes: [input_506, add_54, out_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.rrelu_with_noise_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_rrelu_with_noise_functional_29.run(buf629, buf626, primals_852, primals_853, primals_854, primals_855, buf619, buf628, 65536, grid=grid(65536), stream=stream0)
        del primals_855
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, buf61, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (4, 4, 64, 64), (16384, 1, 256, 4))
        buf631 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_30.run(buf630, buf631, 16, 4096, grid=grid(16, 4096), stream=stream0)
        del buf630
    return (buf631, buf0, buf1, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, buf2, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, buf3, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_46, buf4, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_61, buf5, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, buf6, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, buf7, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_106, buf8, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, buf9, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, buf10, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_151, buf11, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_166, buf12, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, buf13, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, buf14, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_216, buf15, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_227, primals_228, primals_229, primals_230, primals_231, buf16, buf17, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_243, primals_244, primals_245, primals_246, primals_247, buf18, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_258, primals_259, primals_260, primals_261, primals_262, buf19, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_273, primals_274, primals_275, primals_276, primals_277, buf20, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_289, primals_290, primals_291, primals_292, buf21, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_303, primals_304, primals_305, primals_306, primals_307, buf22, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_318, primals_319, primals_320, primals_321, primals_322, buf23, buf24, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_334, primals_335, primals_336, primals_337, primals_338, buf25, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_349, primals_350, primals_351, primals_352, primals_353, buf26, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_364, primals_365, primals_366, primals_367, primals_368, buf27, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_379, primals_380, primals_381, primals_382, primals_383, buf28, buf29, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_395, primals_396, primals_397, primals_398, primals_399, buf30, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_410, primals_411, primals_412, primals_413, primals_414, buf31, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_425, primals_426, primals_427, primals_428, primals_429, buf32, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_440, primals_441, primals_442, primals_443, primals_444, buf33, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_455, primals_456, primals_457, primals_458, primals_459, buf34, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_470, primals_471, primals_472, primals_473, primals_474, buf35, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_485, primals_486, primals_487, primals_488, primals_489, buf36, buf37, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_501, primals_502, primals_503, primals_504, primals_505, buf38, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_516, primals_517, primals_518, primals_519, primals_521, primals_522, primals_523, primals_524, primals_525, buf39, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_536, primals_537, primals_538, primals_539, primals_540, buf40, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_551, primals_552, primals_553, primals_554, primals_555, buf41, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_566, primals_567, primals_568, primals_569, primals_570, buf42, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_581, primals_582, primals_583, primals_584, primals_585, buf43, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_596, primals_597, primals_598, primals_599, primals_600, buf44, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_611, primals_612, primals_613, primals_614, primals_615, buf45, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_626, primals_627, primals_628, primals_629, primals_630, buf46, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_641, primals_642, primals_643, primals_644, primals_645, buf47, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_656, primals_657, primals_658, primals_659, primals_660, buf48, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_671, primals_672, primals_673, primals_674, primals_675, buf49, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_686, primals_687, primals_688, primals_689, primals_691, primals_692, primals_693, primals_694, primals_695, buf50, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_706, primals_707, primals_708, primals_709, primals_710, buf51, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_721, primals_722, primals_723, primals_724, primals_725, buf52, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_736, primals_737, primals_738, primals_739, primals_740, buf53, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_751, primals_752, primals_753, primals_754, primals_755, buf54, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_766, primals_767, primals_768, primals_769, primals_770, buf55, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_781, primals_782, primals_783, primals_784, primals_785, buf56, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_796, primals_797, primals_798, primals_799, primals_800, buf57, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_811, primals_812, primals_813, primals_814, primals_815, buf58, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_826, primals_827, primals_828, primals_829, primals_830, buf59, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_841, primals_842, primals_843, primals_844, primals_845, buf60, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, buf61, buf63, buf65, buf66, buf67, buf68, buf69, buf71, buf72, buf74, buf75, buf77, buf78, buf80, buf81, buf83, buf84, buf86, buf87, buf88, buf90, buf91, buf93, buf94, buf96, buf97, buf98, buf100, buf101, buf103, buf104, buf106, buf107, buf108, buf110, buf111, buf113, buf114, buf116, buf117, buf118, buf120, buf121, buf123, buf124, buf126, buf127, buf128, buf130, buf131, buf133, buf134, buf136, buf137, buf138, buf140, buf141, buf143, buf144, buf146, buf147, buf148, buf150, buf151, buf153, buf154, buf156, buf157, buf158, buf160, buf161, buf163, buf164, buf166, buf167, buf168, buf170, buf171, buf173, buf174, buf176, buf177, buf178, buf179, buf180, buf181, buf183, buf184, buf186, buf187, buf189, buf190, buf192, buf193, buf195, buf196, buf198, buf199, buf200, buf202, buf203, buf205, buf206, buf208, buf209, buf210, buf212, buf213, buf214, buf216, buf217, buf219, buf220, buf221, buf223, buf224, buf226, buf227, buf229, buf230, buf231, buf233, buf234, buf236, buf237, buf239, buf240, buf241, buf243, buf244, buf246, buf247, buf249, buf250, buf251, buf253, buf254, buf256, buf257, buf259, buf260, buf261, buf263, buf264, buf266, buf267, buf269, buf270, buf271, buf273, buf274, buf275, buf277, buf278, buf280, buf281, buf282, buf284, buf285, buf287, buf288, buf290, buf291, buf292, buf294, buf295, buf297, buf298, buf300, buf301, buf302, buf304, buf305, buf307, buf308, buf310, buf311, buf312, buf314, buf315, buf316, buf318, buf319, buf321, buf322, buf323, buf325, buf326, buf328, buf329, buf331, buf332, buf333, buf335, buf336, buf338, buf339, buf341, buf342, buf343, buf345, buf346, buf348, buf349, buf351, buf352, buf353, buf355, buf356, buf358, buf359, buf361, buf362, buf363, buf365, buf366, buf368, buf369, buf371, buf372, buf373, buf375, buf376, buf378, buf379, buf381, buf382, buf383, buf385, buf386, buf387, buf389, buf390, buf392, buf393, buf394, buf396, buf397, buf399, buf400, buf402, buf403, buf404, buf407, buf409, buf410, buf412, buf413, buf415, buf416, buf417, buf419, buf420, buf422, buf423, buf425, buf426, buf427, buf429, buf430, buf432, buf433, buf435, buf436, buf437, buf439, buf440, buf442, buf443, buf445, buf446, buf447, buf449, buf450, buf452, buf453, buf455, buf456, buf457, buf459, buf460, buf462, buf463, buf465, buf466, buf467, buf469, buf470, buf472, buf473, buf475, buf476, buf477, buf479, buf480, buf482, buf483, buf485, buf486, buf487, buf489, buf490, buf492, buf493, buf495, buf496, buf497, buf499, buf500, buf502, buf503, buf505, buf506, buf507, buf509, buf510, buf512, buf513, buf515, buf516, buf517, buf520, buf522, buf523, buf525, buf526, buf528, buf529, buf530, buf532, buf533, buf535, buf536, buf538, buf539, buf540, buf542, buf543, buf545, buf546, buf548, buf549, buf550, buf552, buf553, buf555, buf556, buf558, buf559, buf560, buf562, buf563, buf565, buf566, buf568, buf569, buf570, buf572, buf573, buf575, buf576, buf578, buf579, buf580, buf582, buf583, buf585, buf586, buf588, buf589, buf590, buf592, buf593, buf595, buf596, buf598, buf599, buf600, buf602, buf603, buf605, buf606, buf608, buf609, buf610, buf612, buf613, buf615, buf616, buf618, buf619, buf620, buf622, buf623, buf625, buf626, buf628, buf629, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((13, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, 16, 2, 2), (64, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((32, 32, 2, 2), (128, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((32, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((32, 32, 1, 5), (160, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((32, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((32, 32, 1, 5), (160, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((32, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((32, 32, 1, 5), (160, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((32, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((32, 32, 1, 5), (160, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((16, 16, 2, 2), (64, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((4, 4, 2, 2), (16, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((16, 4, 2, 2), (16, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
