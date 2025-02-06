# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/e3/ce3u7faaeri2rxxadktfkexavnkddk5loebb4t7kncw6odln7sm7.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/qv/cqvb63i7avfibbermmm7lhfggguhil6x4d62bmuporynvqhorf6s.py
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
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 80)
    y1 = yindex // 80
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 80*x2 + 720*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/44/c44l3tf33texdlj5mmwati3oyu5hpma26of7tt6bw7pl7islurws.py
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
    size_hints={'y': 4096, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 48)
    y1 = yindex // 48
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 48*x2 + 1200*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uz/cuzirqw7rrhifo2gtyqo6gghweafsjspqwhzqigiv2s34b5jxuux.py
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
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
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


# kernel path: inductor_cache/36/c36dgp7ez5tyull6il47qdvjxbx6wkvx7ldi2mrbecqjfnlz6jcg.py
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
    ynumel = 9216
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3m/c3mcx4sdbarusyus2hyxjtye2w7y7ftv6hqyun2s4qn5qayuo6xk.py
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
    ynumel = 110592
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 288)
    y1 = yindex // 288
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 288*x2 + 2592*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/s2/cs2f62bbvkttrcvvrouwvyjorogilyibvaklbh7j2y73ucmflyq2.py
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
    size_hints={'y': 16384, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 896*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u7/cu7uewgw5z3y3m2nbfofpnrd5ihp2tprfdip57t5id3zubvaiynv.py
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
    size_hints={'y': 32768, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 896*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jr/cjroqtcenaluo4wx3gsbhyfpdjo6542fdaj5evtzjxyagkmcwyfb.py
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
    size_hints={'y': 32768, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 160)
    y1 = yindex // 160
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 160*x2 + 1120*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lu/clusd2bkz572c47mvgyme7i5qbwubpqclaejcsjgahfqdmr5op6k.py
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
    size_hints={'y': 32768, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30720
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 160)
    y1 = yindex // 160
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 160*x2 + 1120*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yj/cyjed3arrrkjny6hx42i2j5ibiphh3ppidvhbbjdoeg3csinmqhu.py
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
    size_hints={'y': 65536, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1344*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2a/c2akcbr4fkdzntb2eohxj7gcfl7tygauci4sgq2gqrvyuflzkial.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_13 = async_compile.triton('triton_poi_fused_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 61440
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


# kernel path: inductor_cache/pk/cpkpo3kydviber5aed5qhxecpi57gpetkdiwauyupdh7sctrujur.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_14 = async_compile.triton('triton_poi_fused_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ub/cubf3otcl577dwgpvp54hqxk6kqsrca55ja6h54ar2hcrmqt3aal.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_15 = async_compile.triton('triton_poi_fused_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147456
    xnumel = 3
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
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 1152*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/dv/cdv7vs5wktuuyix5sdp2ieot2w2hxf3z2y3efpwfknc6hofeqohr.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_16 = async_compile.triton('triton_poi_fused_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 172032
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 448)
    y1 = yindex // 448
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 448*x2 + 4032*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/mi/cmiagf4s5sbyhsfolbijezljwumymndxcj7gveb4dqbeddwyrdr3.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add, %add_1, %add_2], 1), kwargs = {})
triton_poi_fused_cat_17 = async_compile.triton('triton_poi_fused_cat_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 3)
    x1 = ((xindex // 3) % 16384)
    x2 = xindex // 49152
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + 65536*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.458
    tmp7 = tmp5 * tmp6
    tmp8 = -0.030000000000000027
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr0 + (16384 + x1 + 65536*x2), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = 0.448
    tmp18 = tmp16 * tmp17
    tmp19 = -0.08799999999999997
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp23 = tmp0 >= tmp13
    tmp24 = tl.full([1], 3, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr0 + (32768 + x1 + 65536*x2), tmp23, eviction_policy='evict_last', other=0.0)
    tmp27 = 0.45
    tmp28 = tmp26 * tmp27
    tmp29 = -0.18799999999999994
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp23, tmp30, tmp31)
    tmp33 = tl.where(tmp15, tmp22, tmp32)
    tmp34 = tl.where(tmp4, tmp11, tmp33)
    tl.store(out_ptr0 + (x3), tmp34, None)
''', device_str='cuda')


# kernel path: inductor_cache/oo/coocfr5m76tjzx3pavizfsnbvzuxhdtpetw7glpgj7qrhf6ftpdx.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_2 => add_4, mul_4, mul_5, sub
#   x_3 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_4), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_6), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_8), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_10), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 508032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/6q/c6qhpokgctyy4kn62cgpfnml3xdswlpyh4boig6jh3vrkgvubfpi.py
# Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_5 => add_6, mul_7, mul_8, sub_1
#   x_6 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_12), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_14), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_16), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_18), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/iv/civeny7vh2tvhydn3lou4536qggjpze4jvnrf5xd733i6c4dwhia.py
# Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_8 => add_8, mul_10, mul_11, sub_2
#   x_9 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_20), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_22), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_24), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_26), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 952576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/b2/cb2l7o5rys6d4o6rdk42pi2f7yf4ofjfvqeahnuw3lyvjsvfkxqc.py
# Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_10 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_21 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_21(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 30)
    x2 = ((xindex // 1920) % 30)
    x3 = xindex // 57600
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 7808*x2 + 238144*x3), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 7808*x2 + 238144*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 128*x1 + 7808*x2 + 238144*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (3904 + x0 + 128*x1 + 7808*x2 + 238144*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (3968 + x0 + 128*x1 + 7808*x2 + 238144*x3), xmask)
    tmp9 = tl.load(in_ptr0 + (4032 + x0 + 128*x1 + 7808*x2 + 238144*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (7808 + x0 + 128*x1 + 7808*x2 + 238144*x3), xmask)
    tmp13 = tl.load(in_ptr0 + (7872 + x0 + 128*x1 + 7808*x2 + 238144*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (7936 + x0 + 128*x1 + 7808*x2 + 238144*x3), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sp/csp6rup3dixfhrltv72syri3vwyvax5dzyqbrb5rmtxnz5eyxkfa.py
# Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_12 => add_10, mul_13, mul_14, sub_3
#   x_13 => relu_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_28), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_30), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_32), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_34), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 80)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/2z/c2zaul7y6bd5ekhlq6fohvkscekfthnbgnev6rhuclvebo5325ba.py
# Topologically Sorted Source Nodes: [x_15, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_15 => add_12, mul_16, mul_17, sub_4
#   x_16 => relu_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_36), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_38), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_40), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_42), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/dj/cdjyuecdk6veja557vioy3n7wka2qwnzfjrnd3iecaiz7m6gy2h3.py
# Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_17 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=5] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_24 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_24(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 13)
    x2 = ((xindex // 2496) % 13)
    x3 = xindex // 32448
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1 + 10752*x2 + 150528*x3), xmask)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + 384*x1 + 10752*x2 + 150528*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (384 + x0 + 384*x1 + 10752*x2 + 150528*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (5376 + x0 + 384*x1 + 10752*x2 + 150528*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (5568 + x0 + 384*x1 + 10752*x2 + 150528*x3), xmask)
    tmp9 = tl.load(in_ptr0 + (5760 + x0 + 384*x1 + 10752*x2 + 150528*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (10752 + x0 + 384*x1 + 10752*x2 + 150528*x3), xmask)
    tmp13 = tl.load(in_ptr0 + (10944 + x0 + 384*x1 + 10752*x2 + 150528*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (11136 + x0 + 384*x1 + 10752*x2 + 150528*x3), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4u/c4u5nsbrkdhsnp45oh7ngmhhi5dpzz2zqn3oxycf2qlxagmeer33.py
# Topologically Sorted Source Nodes: [x_21, branch5x5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch5x5 => relu_6
#   x_21 => add_16, mul_22, mul_23, sub_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_52), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_54), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_56), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_58), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 48)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/hd/chdeoqw7d47qbyyeevtuvp6t5qj5luoewp2jro5oirn2l2sisqrb.py
# Topologically Sorted Source Nodes: [x_25, branch3x3dbl], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl => relu_8
#   x_25 => add_20, mul_28, mul_29, sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_68), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_70), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_72), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_74), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_20,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/ud/cudhmohb5terwcnp3bbs34kgzjveu6nbs4chsygcyye2rjcu2mz3.py
# Topologically Sorted Source Nodes: [x_27, branch3x3dbl_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl_1 => relu_9
#   x_27 => add_22, mul_31, mul_32, sub_9
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_76), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_78), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_80), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_82), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_22,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/gj/cgjxzsaiuxwlbyoi5f34a4jny6pamdzqish7j7zylchr3fbxrsxb.py
# Topologically Sorted Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_2, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_28 = async_compile.triton('triton_poi_fused_avg_pool2d_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 2496) % 13)
    x1 = ((xindex // 192) % 13)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 13, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-2688) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-2496) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-2304) + x6), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-192) + x6), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (192 + x6), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (2304 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (2496 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (2688 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14)))*((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14))) + ((-1)*x1*((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14)))) + ((-1)*x2*((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14)))) + ((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14))) + ((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nh/cnhv635fuh4yvpgltbuhmvbkap536fhwa4fhtgy7q4lrikwgc4fa.py
# Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_32 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_5, %relu_7, %relu_10, %relu_11], 1), kwargs = {})
triton_poi_fused_cat_29 = async_compile.triton('triton_poi_fused_cat_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 173056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 128, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (64*x1 + ((-64) + x0)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 224, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (96*x1 + ((-128) + x0)), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = 0.001
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 256, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (32*x1 + ((-224) + x0)), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x2), tmp98, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3e/c3eznn5oesseu7xfafi73kq5hasnbrd5tyopjj5xrq4dh6ocj536.py
# Topologically Sorted Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_2 => avg_pool2d_1
# Graph fragment:
#   %avg_pool2d_1 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_1, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_30 = async_compile.triton('triton_poi_fused_avg_pool2d_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 173056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 3328) % 13)
    x1 = ((xindex // 256) % 13)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 13, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-3584) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-3328) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3072) + x6), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-256) + x6), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (256 + x6), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3072 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (3328 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (3584 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14)))*((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14))) + ((-1)*x1*((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14)))) + ((-1)*x2*((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14)))) + ((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14))) + ((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p5/cp525lk27c4waozecj64ieeh7bg44e6znjblz23jeypavisqlqto.py
# Topologically Sorted Source Nodes: [x_47], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_47 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %relu_14, %relu_17, %relu_18], 1), kwargs = {})
triton_poi_fused_cat_31 = async_compile.triton('triton_poi_fused_cat_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 194688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 288)
    x1 = xindex // 288
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 128, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (64*x1 + ((-64) + x0)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 224, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (96*x1 + ((-128) + x0)), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = 0.001
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 288, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (64*x1 + ((-224) + x0)), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x2), tmp98, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bs/cbswfgabsj2xl72kwvxpmgywhjqujsgbd45szpbq7guu6r542noq.py
# Topologically Sorted Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_4 => avg_pool2d_2
# Graph fragment:
#   %avg_pool2d_2 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_2, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_32 = async_compile.triton('triton_poi_fused_avg_pool2d_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 194688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 3744) % 13)
    x1 = ((xindex // 288) % 13)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 13, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-4032) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-3744) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3456) + x6), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-288) + x6), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (288 + x6), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3456 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (3744 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (4032 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14)))*((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14))) + ((-1)*x1*((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14)))) + ((-1)*x2*((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14)))) + ((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14))) + ((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ji/cjik3ddoqf45vho2iaxatf43ndvyqa6dvruqtg35us3w5f7l5yzb.py
# Topologically Sorted Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   branch_pool_6 => _low_memory_max_pool2d_with_offsets_2, getitem_5
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_3, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_33 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_33(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 144
    xnumel = 288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 6)
    y1 = ((yindex // 6) % 6)
    y2 = yindex // 36
    y4 = (yindex % 36)
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + 576*y0 + 7488*y1 + 48672*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (288 + x3 + 576*y0 + 7488*y1 + 48672*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (576 + x3 + 576*y0 + 7488*y1 + 48672*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3744 + x3 + 576*y0 + 7488*y1 + 48672*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4032 + x3 + 576*y0 + 7488*y1 + 48672*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4320 + x3 + 576*y0 + 7488*y1 + 48672*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (7488 + x3 + 576*y0 + 7488*y1 + 48672*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7776 + x3 + 576*y0 + 7488*y1 + 48672*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8064 + x3 + 576*y0 + 7488*y1 + 48672*y2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1, 1], 1, tl.int8)
    tmp19 = tl.full([1, 1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1, 1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1, 1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1, 1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1, 1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1, 1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1, 1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1, 1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (y4 + 36*x3 + 27648*y2), tmp16, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 288*y5), tmp41, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/5z/c5zbespdxujjghlgqlbb7k6i5x37axqdt2a7knwsbbyirkilh74w.py
# Topologically Sorted Source Nodes: [x_64, branch3x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3 => relu_26
#   x_64 => add_56, mul_82, mul_83, sub_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_212), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_214), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_216), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_218), kwargs = {})
#   %relu_26 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (y0 + 384*x2 + 13824*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2 + 36*y0 + 27648*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/6w/c6wnxzwswh5gsjtr6dy4k4bzxv3u77g2ybxdey35fgzizjyfojmv.py
# Topologically Sorted Source Nodes: [x_70, branch3x3dbl_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl_11 => relu_29
#   x_70 => add_62, mul_91, mul_92, sub_29
# Graph fragment:
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_236), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_238), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_240), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_242), kwargs = {})
#   %relu_29 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_62,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 3456*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2 + 36*y0 + 27648*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/6w/c6watjsfldtafpzo4tnz3ydturqihomornnk6agzyfigildrdnmt.py
# Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_71 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_26, %relu_29, %getitem_4], 1), kwargs = {})
triton_poi_fused_cat_36 = async_compile.triton('triton_poi_fused_cat_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 768)
    y1 = yindex // 768
    tmp0 = tl.load(in_ptr0 + (x2 + 36*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 768*x2 + 27648*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wr/cwrhnzdbydlpbzefhj3qnvklvnsea7lyax42uhnqfo5pxlmrn7be.py
# Topologically Sorted Source Nodes: [x_75, branch7x7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7 => relu_31
#   x_75 => add_66, mul_97, mul_98, sub_31
# Graph fragment:
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_252), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_254), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_256), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_258), kwargs = {})
#   %relu_31 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_66,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
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
    tmp4 = 0.001
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


# kernel path: inductor_cache/v7/cv7zcw6pb2ksjwbrqwfgvfe4kyfoalupjfchaskooxdyevd6mjwe.py
# Topologically Sorted Source Nodes: [branch_pool_7], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_7 => avg_pool2d_3
# Graph fragment:
#   %avg_pool2d_3 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_4, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_38 = async_compile.triton('triton_poi_fused_avg_pool2d_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 110592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4608) % 6)
    x1 = ((xindex // 768) % 6)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5376) + x6), tmp10, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4608) + x6), tmp16, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3840) + x6), tmp23, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-768) + x6), tmp30, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (768 + x6), tmp36, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3840 + x6), tmp43, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4608 + x6), tmp46, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5376 + x6), tmp49, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((7) * ((7) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (7)))*((7) * ((7) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (7))) + ((-1)*x1*((7) * ((7) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (7)))) + ((-1)*x2*((7) * ((7) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (7)))) + ((7) * ((7) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (7))) + ((7) * ((7) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (7)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/wn/cwnzers5qb7mcpizjth2l55fmhpm2vdb66xn5636npr5xktvjqfl.py
# Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_92 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_30, %relu_33, %relu_38, %relu_39], 1), kwargs = {})
triton_poi_fused_cat_39 = async_compile.triton('triton_poi_fused_cat_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 110592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = xindex // 768
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (192*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 384, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (192*x1 + ((-192) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-192) + x0), tmp28, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-192) + x0), tmp28, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-192) + x0), tmp28, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-192) + x0), tmp28, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 576, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (192*x1 + ((-384) + x0)), tmp52, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-384) + x0), tmp52, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-384) + x0), tmp52, eviction_policy='evict_last', other=0.0)
    tmp57 = 0.001
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-384) + x0), tmp52, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-384) + x0), tmp52, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 768, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (192*x1 + ((-576) + x0)), tmp73, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-576) + x0), tmp73, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-576) + x0), tmp73, eviction_policy='evict_last', other=0.0)
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-576) + x0), tmp73, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-576) + x0), tmp73, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x2), tmp98, None)
''', device_str='cuda')


# kernel path: inductor_cache/fx/cfx4666bmskauvhb6jcf3zqzn2tnmlzxbj6362f4fczgoisb7z5y.py
# Topologically Sorted Source Nodes: [x_96, branch7x7_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7_3 => relu_41
#   x_96 => add_86, mul_127, mul_128, sub_41
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_332), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_334), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_336), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_338), kwargs = {})
#   %relu_41 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_86,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/sp/csp5a77d4zfzjnt6g62ylysbywe5jqq6hf2uuurd2j2peiekgjr2.py
# Topologically Sorted Source Nodes: [x_138, branch7x7_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7_9 => relu_61
#   x_138 => add_126, mul_187, mul_188, sub_61
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_492), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_494), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_187, %unsqueeze_496), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %unsqueeze_498), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_126,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/wn/cwnd6u37dv6kl6n7ugdargbrkq4q7kmkvwvee6mbmwrlwmhopcjb.py
# Topologically Sorted Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   branch_pool_15 => _low_memory_max_pool2d_with_offsets_3, getitem_7
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_8, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_42 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_42(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 2)
    y1 = ((yindex // 2) % 2)
    y2 = yindex // 4
    y4 = (yindex % 4)
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + 1536*y0 + 9216*y1 + 27648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (768 + x3 + 1536*y0 + 9216*y1 + 27648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1536 + x3 + 1536*y0 + 9216*y1 + 27648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4608 + x3 + 1536*y0 + 9216*y1 + 27648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (5376 + x3 + 1536*y0 + 9216*y1 + 27648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (6144 + x3 + 1536*y0 + 9216*y1 + 27648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (9216 + x3 + 1536*y0 + 9216*y1 + 27648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (9984 + x3 + 1536*y0 + 9216*y1 + 27648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (10752 + x3 + 1536*y0 + 9216*y1 + 27648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1, 1], 1, tl.int8)
    tmp19 = tl.full([1, 1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1, 1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1, 1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1, 1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1, 1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1, 1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1, 1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1, 1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (y4 + 4*x3 + 5120*y2), tmp16, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 768*y5), tmp41, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/hg/chg3zrmmyyfaxn2tgitbndjyj7j5dwd5iabudfghvypk3vhbnluh.py
# Topologically Sorted Source Nodes: [x_159, branch3x3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3_2 => relu_71
#   x_159 => add_146, mul_217, mul_218, sub_71
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_572), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_574), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_576), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_578), kwargs = {})
#   %relu_71 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_146,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (y0 + 320*x2 + 1280*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2 + 4*y0 + 5120*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/u6/cu6gxsjeaogu5jifwqehpeir5mnn26n4zlpxcb2r5nmx4tthtvt4.py
# Topologically Sorted Source Nodes: [x_167, branch7x7x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7x3_3 => relu_75
#   x_167 => add_154, mul_229, mul_230, sub_75
# Graph fragment:
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_75, %unsqueeze_604), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %unsqueeze_606), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_229, %unsqueeze_608), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_230, %unsqueeze_610), kwargs = {})
#   %relu_75 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_154,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 768*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2 + 4*y0 + 5120*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zz/czzjmldo2n2uk3klyvkoxlvxtw363oqrrdda3nc5vsdgopqscodp.py
# Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_168 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_71, %relu_75, %getitem_6], 1), kwargs = {})
triton_poi_fused_cat_45 = async_compile.triton('triton_poi_fused_cat_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_45(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1280)
    y1 = yindex // 1280
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1280*x2 + 5120*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xf/cxfedcwdrrgzclmsqpp45j2rlt5jnm3mkwlkpjmmj62ic4mtnaem.py
# Topologically Sorted Source Nodes: [x_172, branch3x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3_3 => relu_77
#   x_172 => add_158, mul_235, mul_236, sub_77
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_620), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_622), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %unsqueeze_624), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %unsqueeze_626), kwargs = {})
#   %relu_77 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_158,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/sg/csgwipk7i5jxh3ridszxjmjri7dfyetkzxuvliwlr2bjocl5caoh.py
# Topologically Sorted Source Nodes: [branch3x3_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   branch3x3_4 => cat_10
# Graph fragment:
#   %cat_10 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_78, %relu_79], 1), kwargs = {})
triton_poi_fused_cat_47 = async_compile.triton('triton_poi_fused_cat_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 768)
    x0 = (xindex % 4)
    x2 = xindex // 3072
    x3 = (xindex % 3072)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (384*x0 + 1536*x2 + (x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 768, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (384*x0 + 1536*x2 + ((-384) + x1)), tmp25, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-384) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-384) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-384) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-384) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3 + 8192*x2), tmp48, None)
''', device_str='cuda')


# kernel path: inductor_cache/rg/crgyxgd4xpwpdg6aspadbksdoy5ex47765wdu5rrwhv3pv6glafv.py
# Topologically Sorted Source Nodes: [x_178, branch3x3dbl_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl_12 => relu_80
#   x_178 => add_164, mul_244, mul_245, sub_80
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_644), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_646), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_244, %unsqueeze_648), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_245, %unsqueeze_650), kwargs = {})
#   %relu_80 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_164,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 448)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/k3/ck3p23boblwrjpj3jfdkpabhdgc4mtygf4mxc5ddlqp6fqu5v6y4.py
# Topologically Sorted Source Nodes: [branch_pool_16], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_16 => avg_pool2d_7
# Graph fragment:
#   %avg_pool2d_7 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_9, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_49 = async_compile.triton('triton_poi_fused_avg_pool2d_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 2560) % 2)
    x1 = ((xindex // 1280) % 2)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-3840) + x6), tmp10, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-2560) + x6), tmp16, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-1280) + x6), tmp23, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1280) + x6), tmp30, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1280 + x6), tmp36, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (1280 + x6), tmp43, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (2560 + x6), tmp46, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (3840 + x6), tmp49, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((3) * ((3) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (3)))*((3) * ((3) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (3))) + ((-1)*x1*((3) * ((3) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (3)))) + ((-1)*x2*((3) * ((3) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (3)))) + ((3) * ((3) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (3))) + ((3) * ((3) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (3)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/f6/cf6k4ecfwdgqpbzrtnnsmmnuillilzlbpoug64qv47bkmgnqg7uo.py
# Topologically Sorted Source Nodes: [x_170, branch1x1_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch1x1_7 => relu_76
#   x_170 => add_156, mul_232, mul_233, sub_76
# Graph fragment:
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_612), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_614), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_232, %unsqueeze_616), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_233, %unsqueeze_618), kwargs = {})
#   %relu_76 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_156,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_50', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (y0 + 320*x2 + 1280*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2 + 4*y0 + 8192*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/vn/cvnsnaczvykastwrjbwwkagwxrousbvi3xczsthdthnwnxcpylsn.py
# Topologically Sorted Source Nodes: [x_186, branch_pool_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch_pool_17 => relu_84
#   x_186 => add_172, mul_256, mul_257, sub_84
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_676), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_678), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_680), kwargs = {})
#   %add_172 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_682), kwargs = {})
#   %relu_84 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_172,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_51', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 768*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2 + 4*y0 + 8192*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/lh/clhieoe7o2aawrg4cm77gslefkd7pm4ik4xemkksumlqqxinjlwc.py
# Topologically Sorted Source Nodes: [x_187], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_187 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_76, %cat_10, %cat_11, %relu_84], 1), kwargs = {})
triton_poi_fused_cat_52 = async_compile.triton('triton_poi_fused_cat_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_52(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 2048*x2 + 8192*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jk/cjklql5b5flplr54xfsehrrwjbzay4c6fekda3i7ut7nksf3brcz.py
# Topologically Sorted Source Nodes: [branch_pool_18], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_18 => avg_pool2d_8
# Graph fragment:
#   %avg_pool2d_8 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_12, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_53 = async_compile.triton('triton_poi_fused_avg_pool2d_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4096) % 2)
    x1 = ((xindex // 2048) % 2)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6144) + x6), tmp10, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4096) + x6), tmp16, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-2048) + x6), tmp23, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-2048) + x6), tmp30, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (2048 + x6), tmp36, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (2048 + x6), tmp43, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4096 + x6), tmp46, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (6144 + x6), tmp49, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((3) * ((3) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (3)))*((3) * ((3) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (3))) + ((-1)*x1*((3) * ((3) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (3)))) + ((-1)*x2*((3) * ((3) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (3)))) + ((3) * ((3) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (3))) + ((3) * ((3) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (3)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/ib/cibywekgqittu455dh5j74b7knf3mjd74bk7j6drb2rgwihvf37w.py
# Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_207 => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_15, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_54 = async_compile.triton('triton_poi_fused_mean_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_54(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 128, 128), (65536, 16384, 128, 1))
    assert_size_stride(primals_2, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, ), (1, ))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_18, (80, ), (1, ))
    assert_size_stride(primals_19, (80, ), (1, ))
    assert_size_stride(primals_20, (80, ), (1, ))
    assert_size_stride(primals_21, (80, ), (1, ))
    assert_size_stride(primals_22, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_23, (192, ), (1, ))
    assert_size_stride(primals_24, (192, ), (1, ))
    assert_size_stride(primals_25, (192, ), (1, ))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_33, (48, ), (1, ))
    assert_size_stride(primals_34, (48, ), (1, ))
    assert_size_stride(primals_35, (48, ), (1, ))
    assert_size_stride(primals_36, (48, ), (1, ))
    assert_size_stride(primals_37, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_48, (96, ), (1, ))
    assert_size_stride(primals_49, (96, ), (1, ))
    assert_size_stride(primals_50, (96, ), (1, ))
    assert_size_stride(primals_51, (96, ), (1, ))
    assert_size_stride(primals_52, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_53, (96, ), (1, ))
    assert_size_stride(primals_54, (96, ), (1, ))
    assert_size_stride(primals_55, (96, ), (1, ))
    assert_size_stride(primals_56, (96, ), (1, ))
    assert_size_stride(primals_57, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_58, (32, ), (1, ))
    assert_size_stride(primals_59, (32, ), (1, ))
    assert_size_stride(primals_60, (32, ), (1, ))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_68, (48, ), (1, ))
    assert_size_stride(primals_69, (48, ), (1, ))
    assert_size_stride(primals_70, (48, ), (1, ))
    assert_size_stride(primals_71, (48, ), (1, ))
    assert_size_stride(primals_72, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_83, (96, ), (1, ))
    assert_size_stride(primals_84, (96, ), (1, ))
    assert_size_stride(primals_85, (96, ), (1, ))
    assert_size_stride(primals_86, (96, ), (1, ))
    assert_size_stride(primals_87, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_88, (96, ), (1, ))
    assert_size_stride(primals_89, (96, ), (1, ))
    assert_size_stride(primals_90, (96, ), (1, ))
    assert_size_stride(primals_91, (96, ), (1, ))
    assert_size_stride(primals_92, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, ), (1, ))
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_96, (64, ), (1, ))
    assert_size_stride(primals_97, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_98, (64, ), (1, ))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (64, ), (1, ))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_103, (48, ), (1, ))
    assert_size_stride(primals_104, (48, ), (1, ))
    assert_size_stride(primals_105, (48, ), (1, ))
    assert_size_stride(primals_106, (48, ), (1, ))
    assert_size_stride(primals_107, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_108, (64, ), (1, ))
    assert_size_stride(primals_109, (64, ), (1, ))
    assert_size_stride(primals_110, (64, ), (1, ))
    assert_size_stride(primals_111, (64, ), (1, ))
    assert_size_stride(primals_112, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_113, (64, ), (1, ))
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_116, (64, ), (1, ))
    assert_size_stride(primals_117, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_118, (96, ), (1, ))
    assert_size_stride(primals_119, (96, ), (1, ))
    assert_size_stride(primals_120, (96, ), (1, ))
    assert_size_stride(primals_121, (96, ), (1, ))
    assert_size_stride(primals_122, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_123, (96, ), (1, ))
    assert_size_stride(primals_124, (96, ), (1, ))
    assert_size_stride(primals_125, (96, ), (1, ))
    assert_size_stride(primals_126, (96, ), (1, ))
    assert_size_stride(primals_127, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_128, (64, ), (1, ))
    assert_size_stride(primals_129, (64, ), (1, ))
    assert_size_stride(primals_130, (64, ), (1, ))
    assert_size_stride(primals_131, (64, ), (1, ))
    assert_size_stride(primals_132, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, ), (1, ))
    assert_size_stride(primals_137, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_138, (64, ), (1, ))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, ), (1, ))
    assert_size_stride(primals_141, (64, ), (1, ))
    assert_size_stride(primals_142, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_143, (96, ), (1, ))
    assert_size_stride(primals_144, (96, ), (1, ))
    assert_size_stride(primals_145, (96, ), (1, ))
    assert_size_stride(primals_146, (96, ), (1, ))
    assert_size_stride(primals_147, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_148, (96, ), (1, ))
    assert_size_stride(primals_149, (96, ), (1, ))
    assert_size_stride(primals_150, (96, ), (1, ))
    assert_size_stride(primals_151, (96, ), (1, ))
    assert_size_stride(primals_152, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_153, (192, ), (1, ))
    assert_size_stride(primals_154, (192, ), (1, ))
    assert_size_stride(primals_155, (192, ), (1, ))
    assert_size_stride(primals_156, (192, ), (1, ))
    assert_size_stride(primals_157, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (128, ), (1, ))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_167, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(primals_168, (192, ), (1, ))
    assert_size_stride(primals_169, (192, ), (1, ))
    assert_size_stride(primals_170, (192, ), (1, ))
    assert_size_stride(primals_171, (192, ), (1, ))
    assert_size_stride(primals_172, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (128, ), (1, ))
    assert_size_stride(primals_176, (128, ), (1, ))
    assert_size_stride(primals_177, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (128, ), (1, ))
    assert_size_stride(primals_190, (128, ), (1, ))
    assert_size_stride(primals_191, (128, ), (1, ))
    assert_size_stride(primals_192, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_193, (192, ), (1, ))
    assert_size_stride(primals_194, (192, ), (1, ))
    assert_size_stride(primals_195, (192, ), (1, ))
    assert_size_stride(primals_196, (192, ), (1, ))
    assert_size_stride(primals_197, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_198, (192, ), (1, ))
    assert_size_stride(primals_199, (192, ), (1, ))
    assert_size_stride(primals_200, (192, ), (1, ))
    assert_size_stride(primals_201, (192, ), (1, ))
    assert_size_stride(primals_202, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_203, (192, ), (1, ))
    assert_size_stride(primals_204, (192, ), (1, ))
    assert_size_stride(primals_205, (192, ), (1, ))
    assert_size_stride(primals_206, (192, ), (1, ))
    assert_size_stride(primals_207, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_208, (160, ), (1, ))
    assert_size_stride(primals_209, (160, ), (1, ))
    assert_size_stride(primals_210, (160, ), (1, ))
    assert_size_stride(primals_211, (160, ), (1, ))
    assert_size_stride(primals_212, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_213, (160, ), (1, ))
    assert_size_stride(primals_214, (160, ), (1, ))
    assert_size_stride(primals_215, (160, ), (1, ))
    assert_size_stride(primals_216, (160, ), (1, ))
    assert_size_stride(primals_217, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_218, (192, ), (1, ))
    assert_size_stride(primals_219, (192, ), (1, ))
    assert_size_stride(primals_220, (192, ), (1, ))
    assert_size_stride(primals_221, (192, ), (1, ))
    assert_size_stride(primals_222, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_223, (160, ), (1, ))
    assert_size_stride(primals_224, (160, ), (1, ))
    assert_size_stride(primals_225, (160, ), (1, ))
    assert_size_stride(primals_226, (160, ), (1, ))
    assert_size_stride(primals_227, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_228, (160, ), (1, ))
    assert_size_stride(primals_229, (160, ), (1, ))
    assert_size_stride(primals_230, (160, ), (1, ))
    assert_size_stride(primals_231, (160, ), (1, ))
    assert_size_stride(primals_232, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_233, (160, ), (1, ))
    assert_size_stride(primals_234, (160, ), (1, ))
    assert_size_stride(primals_235, (160, ), (1, ))
    assert_size_stride(primals_236, (160, ), (1, ))
    assert_size_stride(primals_237, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_238, (160, ), (1, ))
    assert_size_stride(primals_239, (160, ), (1, ))
    assert_size_stride(primals_240, (160, ), (1, ))
    assert_size_stride(primals_241, (160, ), (1, ))
    assert_size_stride(primals_242, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_243, (192, ), (1, ))
    assert_size_stride(primals_244, (192, ), (1, ))
    assert_size_stride(primals_245, (192, ), (1, ))
    assert_size_stride(primals_246, (192, ), (1, ))
    assert_size_stride(primals_247, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_248, (192, ), (1, ))
    assert_size_stride(primals_249, (192, ), (1, ))
    assert_size_stride(primals_250, (192, ), (1, ))
    assert_size_stride(primals_251, (192, ), (1, ))
    assert_size_stride(primals_252, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_253, (192, ), (1, ))
    assert_size_stride(primals_254, (192, ), (1, ))
    assert_size_stride(primals_255, (192, ), (1, ))
    assert_size_stride(primals_256, (192, ), (1, ))
    assert_size_stride(primals_257, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_258, (160, ), (1, ))
    assert_size_stride(primals_259, (160, ), (1, ))
    assert_size_stride(primals_260, (160, ), (1, ))
    assert_size_stride(primals_261, (160, ), (1, ))
    assert_size_stride(primals_262, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_263, (160, ), (1, ))
    assert_size_stride(primals_264, (160, ), (1, ))
    assert_size_stride(primals_265, (160, ), (1, ))
    assert_size_stride(primals_266, (160, ), (1, ))
    assert_size_stride(primals_267, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_268, (192, ), (1, ))
    assert_size_stride(primals_269, (192, ), (1, ))
    assert_size_stride(primals_270, (192, ), (1, ))
    assert_size_stride(primals_271, (192, ), (1, ))
    assert_size_stride(primals_272, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_273, (160, ), (1, ))
    assert_size_stride(primals_274, (160, ), (1, ))
    assert_size_stride(primals_275, (160, ), (1, ))
    assert_size_stride(primals_276, (160, ), (1, ))
    assert_size_stride(primals_277, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_278, (160, ), (1, ))
    assert_size_stride(primals_279, (160, ), (1, ))
    assert_size_stride(primals_280, (160, ), (1, ))
    assert_size_stride(primals_281, (160, ), (1, ))
    assert_size_stride(primals_282, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_283, (160, ), (1, ))
    assert_size_stride(primals_284, (160, ), (1, ))
    assert_size_stride(primals_285, (160, ), (1, ))
    assert_size_stride(primals_286, (160, ), (1, ))
    assert_size_stride(primals_287, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_288, (160, ), (1, ))
    assert_size_stride(primals_289, (160, ), (1, ))
    assert_size_stride(primals_290, (160, ), (1, ))
    assert_size_stride(primals_291, (160, ), (1, ))
    assert_size_stride(primals_292, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_293, (192, ), (1, ))
    assert_size_stride(primals_294, (192, ), (1, ))
    assert_size_stride(primals_295, (192, ), (1, ))
    assert_size_stride(primals_296, (192, ), (1, ))
    assert_size_stride(primals_297, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_298, (192, ), (1, ))
    assert_size_stride(primals_299, (192, ), (1, ))
    assert_size_stride(primals_300, (192, ), (1, ))
    assert_size_stride(primals_301, (192, ), (1, ))
    assert_size_stride(primals_302, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_303, (192, ), (1, ))
    assert_size_stride(primals_304, (192, ), (1, ))
    assert_size_stride(primals_305, (192, ), (1, ))
    assert_size_stride(primals_306, (192, ), (1, ))
    assert_size_stride(primals_307, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_308, (192, ), (1, ))
    assert_size_stride(primals_309, (192, ), (1, ))
    assert_size_stride(primals_310, (192, ), (1, ))
    assert_size_stride(primals_311, (192, ), (1, ))
    assert_size_stride(primals_312, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_313, (192, ), (1, ))
    assert_size_stride(primals_314, (192, ), (1, ))
    assert_size_stride(primals_315, (192, ), (1, ))
    assert_size_stride(primals_316, (192, ), (1, ))
    assert_size_stride(primals_317, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_318, (192, ), (1, ))
    assert_size_stride(primals_319, (192, ), (1, ))
    assert_size_stride(primals_320, (192, ), (1, ))
    assert_size_stride(primals_321, (192, ), (1, ))
    assert_size_stride(primals_322, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_323, (192, ), (1, ))
    assert_size_stride(primals_324, (192, ), (1, ))
    assert_size_stride(primals_325, (192, ), (1, ))
    assert_size_stride(primals_326, (192, ), (1, ))
    assert_size_stride(primals_327, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_328, (192, ), (1, ))
    assert_size_stride(primals_329, (192, ), (1, ))
    assert_size_stride(primals_330, (192, ), (1, ))
    assert_size_stride(primals_331, (192, ), (1, ))
    assert_size_stride(primals_332, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_333, (192, ), (1, ))
    assert_size_stride(primals_334, (192, ), (1, ))
    assert_size_stride(primals_335, (192, ), (1, ))
    assert_size_stride(primals_336, (192, ), (1, ))
    assert_size_stride(primals_337, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_338, (192, ), (1, ))
    assert_size_stride(primals_339, (192, ), (1, ))
    assert_size_stride(primals_340, (192, ), (1, ))
    assert_size_stride(primals_341, (192, ), (1, ))
    assert_size_stride(primals_342, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_343, (192, ), (1, ))
    assert_size_stride(primals_344, (192, ), (1, ))
    assert_size_stride(primals_345, (192, ), (1, ))
    assert_size_stride(primals_346, (192, ), (1, ))
    assert_size_stride(primals_347, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_348, (192, ), (1, ))
    assert_size_stride(primals_349, (192, ), (1, ))
    assert_size_stride(primals_350, (192, ), (1, ))
    assert_size_stride(primals_351, (192, ), (1, ))
    assert_size_stride(primals_352, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_353, (192, ), (1, ))
    assert_size_stride(primals_354, (192, ), (1, ))
    assert_size_stride(primals_355, (192, ), (1, ))
    assert_size_stride(primals_356, (192, ), (1, ))
    assert_size_stride(primals_357, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_358, (320, ), (1, ))
    assert_size_stride(primals_359, (320, ), (1, ))
    assert_size_stride(primals_360, (320, ), (1, ))
    assert_size_stride(primals_361, (320, ), (1, ))
    assert_size_stride(primals_362, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_363, (192, ), (1, ))
    assert_size_stride(primals_364, (192, ), (1, ))
    assert_size_stride(primals_365, (192, ), (1, ))
    assert_size_stride(primals_366, (192, ), (1, ))
    assert_size_stride(primals_367, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_368, (192, ), (1, ))
    assert_size_stride(primals_369, (192, ), (1, ))
    assert_size_stride(primals_370, (192, ), (1, ))
    assert_size_stride(primals_371, (192, ), (1, ))
    assert_size_stride(primals_372, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_373, (192, ), (1, ))
    assert_size_stride(primals_374, (192, ), (1, ))
    assert_size_stride(primals_375, (192, ), (1, ))
    assert_size_stride(primals_376, (192, ), (1, ))
    assert_size_stride(primals_377, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_378, (192, ), (1, ))
    assert_size_stride(primals_379, (192, ), (1, ))
    assert_size_stride(primals_380, (192, ), (1, ))
    assert_size_stride(primals_381, (192, ), (1, ))
    assert_size_stride(primals_382, (320, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_383, (320, ), (1, ))
    assert_size_stride(primals_384, (320, ), (1, ))
    assert_size_stride(primals_385, (320, ), (1, ))
    assert_size_stride(primals_386, (320, ), (1, ))
    assert_size_stride(primals_387, (384, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_388, (384, ), (1, ))
    assert_size_stride(primals_389, (384, ), (1, ))
    assert_size_stride(primals_390, (384, ), (1, ))
    assert_size_stride(primals_391, (384, ), (1, ))
    assert_size_stride(primals_392, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_393, (384, ), (1, ))
    assert_size_stride(primals_394, (384, ), (1, ))
    assert_size_stride(primals_395, (384, ), (1, ))
    assert_size_stride(primals_396, (384, ), (1, ))
    assert_size_stride(primals_397, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_398, (384, ), (1, ))
    assert_size_stride(primals_399, (384, ), (1, ))
    assert_size_stride(primals_400, (384, ), (1, ))
    assert_size_stride(primals_401, (384, ), (1, ))
    assert_size_stride(primals_402, (448, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_403, (448, ), (1, ))
    assert_size_stride(primals_404, (448, ), (1, ))
    assert_size_stride(primals_405, (448, ), (1, ))
    assert_size_stride(primals_406, (448, ), (1, ))
    assert_size_stride(primals_407, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(primals_408, (384, ), (1, ))
    assert_size_stride(primals_409, (384, ), (1, ))
    assert_size_stride(primals_410, (384, ), (1, ))
    assert_size_stride(primals_411, (384, ), (1, ))
    assert_size_stride(primals_412, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_413, (384, ), (1, ))
    assert_size_stride(primals_414, (384, ), (1, ))
    assert_size_stride(primals_415, (384, ), (1, ))
    assert_size_stride(primals_416, (384, ), (1, ))
    assert_size_stride(primals_417, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_418, (384, ), (1, ))
    assert_size_stride(primals_419, (384, ), (1, ))
    assert_size_stride(primals_420, (384, ), (1, ))
    assert_size_stride(primals_421, (384, ), (1, ))
    assert_size_stride(primals_422, (192, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_423, (192, ), (1, ))
    assert_size_stride(primals_424, (192, ), (1, ))
    assert_size_stride(primals_425, (192, ), (1, ))
    assert_size_stride(primals_426, (192, ), (1, ))
    assert_size_stride(primals_427, (320, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_428, (320, ), (1, ))
    assert_size_stride(primals_429, (320, ), (1, ))
    assert_size_stride(primals_430, (320, ), (1, ))
    assert_size_stride(primals_431, (320, ), (1, ))
    assert_size_stride(primals_432, (384, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_433, (384, ), (1, ))
    assert_size_stride(primals_434, (384, ), (1, ))
    assert_size_stride(primals_435, (384, ), (1, ))
    assert_size_stride(primals_436, (384, ), (1, ))
    assert_size_stride(primals_437, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_438, (384, ), (1, ))
    assert_size_stride(primals_439, (384, ), (1, ))
    assert_size_stride(primals_440, (384, ), (1, ))
    assert_size_stride(primals_441, (384, ), (1, ))
    assert_size_stride(primals_442, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_443, (384, ), (1, ))
    assert_size_stride(primals_444, (384, ), (1, ))
    assert_size_stride(primals_445, (384, ), (1, ))
    assert_size_stride(primals_446, (384, ), (1, ))
    assert_size_stride(primals_447, (448, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_448, (448, ), (1, ))
    assert_size_stride(primals_449, (448, ), (1, ))
    assert_size_stride(primals_450, (448, ), (1, ))
    assert_size_stride(primals_451, (448, ), (1, ))
    assert_size_stride(primals_452, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(primals_453, (384, ), (1, ))
    assert_size_stride(primals_454, (384, ), (1, ))
    assert_size_stride(primals_455, (384, ), (1, ))
    assert_size_stride(primals_456, (384, ), (1, ))
    assert_size_stride(primals_457, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_458, (384, ), (1, ))
    assert_size_stride(primals_459, (384, ), (1, ))
    assert_size_stride(primals_460, (384, ), (1, ))
    assert_size_stride(primals_461, (384, ), (1, ))
    assert_size_stride(primals_462, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_463, (384, ), (1, ))
    assert_size_stride(primals_464, (384, ), (1, ))
    assert_size_stride(primals_465, (384, ), (1, ))
    assert_size_stride(primals_466, (384, ), (1, ))
    assert_size_stride(primals_467, (192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_468, (192, ), (1, ))
    assert_size_stride(primals_469, (192, ), (1, ))
    assert_size_stride(primals_470, (192, ), (1, ))
    assert_size_stride(primals_471, (192, ), (1, ))
    assert_size_stride(primals_472, (12, 2048), (2048, 1))
    assert_size_stride(primals_473, (12, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_2, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_7, buf1, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_7
        buf2 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_12, buf2, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_12
        buf3 = empty_strided_cuda((192, 80, 3, 3), (720, 1, 240, 80), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_22, buf3, 15360, 9, grid=grid(15360, 9), stream=stream0)
        del primals_22
        buf4 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_37, buf4, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_37
        buf5 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_47, buf5, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_47
        buf6 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_52, buf6, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_52
        buf7 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_72, buf7, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_72
        buf8 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_82, buf8, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_82
        buf9 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_87, buf9, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_87
        buf10 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_107, buf10, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_107
        buf11 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_117, buf11, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_117
        buf12 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_122, buf12, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_122
        buf13 = empty_strided_cuda((384, 288, 3, 3), (2592, 1, 864, 288), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_132, buf13, 110592, 9, grid=grid(110592, 9), stream=stream0)
        del primals_132
        buf14 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_142, buf14, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_142
        buf15 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_147, buf15, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_147
        buf16 = empty_strided_cuda((128, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_162, buf16, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_162
        buf17 = empty_strided_cuda((192, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_167, buf17, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del primals_167
        buf18 = empty_strided_cuda((128, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_177, buf18, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_177
        buf19 = empty_strided_cuda((128, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_182, buf19, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_182
        buf20 = empty_strided_cuda((128, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_187, buf20, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_187
        buf21 = empty_strided_cuda((192, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_192, buf21, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del primals_192
        buf22 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_212, buf22, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_212
        buf23 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_217, buf23, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_217
        buf24 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_227, buf24, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_227
        buf25 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_232, buf25, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_232
        buf26 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_237, buf26, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_237
        buf27 = empty_strided_cuda((192, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_242, buf27, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_242
        buf28 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_262, buf28, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_262
        buf29 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_267, buf29, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_267
        buf30 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_277, buf30, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_277
        buf31 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_282, buf31, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_282
        buf32 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_287, buf32, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_287
        buf33 = empty_strided_cuda((192, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_292, buf33, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_292
        buf34 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_312, buf34, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_312
        buf35 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_317, buf35, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_317
        buf36 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_327, buf36, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_327
        buf37 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_332, buf37, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_332
        buf38 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_337, buf38, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_337
        buf39 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_342, buf39, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_342
        buf40 = empty_strided_cuda((320, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_357, buf40, 61440, 9, grid=grid(61440, 9), stream=stream0)
        del primals_357
        buf41 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_367, buf41, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_367
        buf42 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_372, buf42, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_372
        buf43 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_377, buf43, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_377
        buf44 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_392, buf44, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_392
        buf45 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_397, buf45, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_397
        buf46 = empty_strided_cuda((384, 448, 3, 3), (4032, 1, 1344, 448), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_407, buf46, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del primals_407
        buf47 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_412, buf47, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_412
        buf48 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_417, buf48, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_417
        buf49 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_437, buf49, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_437
        buf50 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_442, buf50, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_442
        buf51 = empty_strided_cuda((384, 448, 3, 3), (4032, 1, 1344, 448), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_452, buf51, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del primals_452
        buf52 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_457, buf52, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_457
        buf53 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_462, buf53, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_462
        buf54 = empty_strided_cuda((4, 3, 128, 128), (49152, 1, 384, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_17.run(primals_1, buf54, 196608, grid=grid(196608), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 32, 63, 63), (127008, 1, 2016, 32))
        buf56 = empty_strided_cuda((4, 32, 63, 63), (127008, 1, 2016, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf55, primals_3, primals_4, primals_5, primals_6, buf56, 508032, grid=grid(508032), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, buf1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf58 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf57, primals_8, primals_9, primals_10, primals_11, buf58, 476288, grid=grid(476288), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf60 = empty_strided_cuda((4, 64, 61, 61), (238144, 1, 3904, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf59, primals_13, primals_14, primals_15, primals_16, buf60, 952576, grid=grid(952576), stream=stream0)
        del primals_16
        buf61 = empty_strided_cuda((4, 64, 30, 30), (57600, 1, 1920, 64), torch.float32)
        buf62 = empty_strided_cuda((4, 64, 30, 30), (57600, 1, 1920, 64), torch.int8)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_21.run(buf60, buf61, buf62, 230400, grid=grid(230400), stream=stream0)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf61, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 80, 30, 30), (72000, 1, 2400, 80))
        buf64 = empty_strided_cuda((4, 80, 30, 30), (72000, 1, 2400, 80), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf63, primals_18, primals_19, primals_20, primals_21, buf64, 288000, grid=grid(288000), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, buf3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 192, 28, 28), (150528, 1, 5376, 192))
        buf66 = empty_strided_cuda((4, 192, 28, 28), (150528, 1, 5376, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_15, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf65, primals_23, primals_24, primals_25, primals_26, buf66, 602112, grid=grid(602112), stream=stream0)
        del primals_26
        buf67 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
        buf68 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.int8)
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_24.run(buf66, buf67, buf68, 129792, grid=grid(129792), stream=stream0)
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf67, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 64, 13, 13), (10816, 1, 832, 64))
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf67, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 48, 13, 13), (8112, 1, 624, 48))
        buf71 = empty_strided_cuda((4, 48, 13, 13), (8112, 1, 624, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_21, branch5x5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf70, primals_33, primals_34, primals_35, primals_36, buf71, 32448, grid=grid(32448), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, buf4, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 64, 13, 13), (10816, 1, 832, 64))
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf67, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 64, 13, 13), (10816, 1, 832, 64))
        buf74 = empty_strided_cuda((4, 64, 13, 13), (10816, 1, 832, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_25, branch3x3dbl], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf73, primals_43, primals_44, primals_45, primals_46, buf74, 43264, grid=grid(43264), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 96, 13, 13), (16224, 1, 1248, 96))
        buf76 = empty_strided_cuda((4, 96, 13, 13), (16224, 1, 1248, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, branch3x3dbl_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf75, primals_48, primals_49, primals_50, primals_51, buf76, 64896, grid=grid(64896), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 96, 13, 13), (16224, 1, 1248, 96))
        buf78 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_28.run(buf67, buf78, 129792, grid=grid(129792), stream=stream0)
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 32, 13, 13), (5408, 1, 416, 32))
        buf80 = empty_strided_cuda((4, 256, 13, 13), (43264, 1, 3328, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_29.run(buf69, primals_28, primals_29, primals_30, primals_31, buf72, primals_38, primals_39, primals_40, primals_41, buf77, primals_53, primals_54, primals_55, primals_56, buf79, primals_58, primals_59, primals_60, primals_61, buf80, 173056, grid=grid(173056), stream=stream0)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 64, 13, 13), (10816, 1, 832, 64))
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf80, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 48, 13, 13), (8112, 1, 624, 48))
        buf83 = empty_strided_cuda((4, 48, 13, 13), (8112, 1, 624, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_36, branch5x5_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf82, primals_68, primals_69, primals_70, primals_71, buf83, 32448, grid=grid(32448), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, buf7, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 64, 13, 13), (10816, 1, 832, 64))
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf80, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 64, 13, 13), (10816, 1, 832, 64))
        buf86 = empty_strided_cuda((4, 64, 13, 13), (10816, 1, 832, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_40, branch3x3dbl_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf85, primals_78, primals_79, primals_80, primals_81, buf86, 43264, grid=grid(43264), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 96, 13, 13), (16224, 1, 1248, 96))
        buf88 = empty_strided_cuda((4, 96, 13, 13), (16224, 1, 1248, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_42, branch3x3dbl_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf87, primals_83, primals_84, primals_85, primals_86, buf88, 64896, grid=grid(64896), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 96, 13, 13), (16224, 1, 1248, 96))
        buf90 = empty_strided_cuda((4, 256, 13, 13), (43264, 1, 3328, 256), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_30.run(buf80, buf90, 173056, grid=grid(173056), stream=stream0)
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 64, 13, 13), (10816, 1, 832, 64))
        buf92 = empty_strided_cuda((4, 288, 13, 13), (48672, 1, 3744, 288), torch.float32)
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_31.run(buf81, primals_63, primals_64, primals_65, primals_66, buf84, primals_73, primals_74, primals_75, primals_76, buf89, primals_88, primals_89, primals_90, primals_91, buf91, primals_93, primals_94, primals_95, primals_96, buf92, 194688, grid=grid(194688), stream=stream0)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 64, 13, 13), (10816, 1, 832, 64))
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf92, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 48, 13, 13), (8112, 1, 624, 48))
        buf95 = empty_strided_cuda((4, 48, 13, 13), (8112, 1, 624, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_51, branch5x5_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf94, primals_103, primals_104, primals_105, primals_106, buf95, 32448, grid=grid(32448), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, buf10, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 64, 13, 13), (10816, 1, 832, 64))
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf92, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 64, 13, 13), (10816, 1, 832, 64))
        buf98 = empty_strided_cuda((4, 64, 13, 13), (10816, 1, 832, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_55, branch3x3dbl_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf97, primals_113, primals_114, primals_115, primals_116, buf98, 43264, grid=grid(43264), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 96, 13, 13), (16224, 1, 1248, 96))
        buf100 = empty_strided_cuda((4, 96, 13, 13), (16224, 1, 1248, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_57, branch3x3dbl_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf99, primals_118, primals_119, primals_120, primals_121, buf100, 64896, grid=grid(64896), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 96, 13, 13), (16224, 1, 1248, 96))
        buf102 = empty_strided_cuda((4, 288, 13, 13), (48672, 1, 3744, 288), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_32.run(buf92, buf102, 194688, grid=grid(194688), stream=stream0)
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 64, 13, 13), (10816, 1, 832, 64))
        buf104 = empty_strided_cuda((4, 288, 13, 13), (48672, 1, 3744, 288), torch.float32)
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_31.run(buf93, primals_98, primals_99, primals_100, primals_101, buf96, primals_108, primals_109, primals_110, primals_111, buf101, primals_123, primals_124, primals_125, primals_126, buf103, primals_128, primals_129, primals_130, primals_131, buf104, 194688, grid=grid(194688), stream=stream0)
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, buf13, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 384, 6, 6), (13824, 1, 2304, 384))
        # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf104, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 64, 13, 13), (10816, 1, 832, 64))
        buf107 = empty_strided_cuda((4, 64, 13, 13), (10816, 1, 832, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_66, branch3x3dbl_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf106, primals_138, primals_139, primals_140, primals_141, buf107, 43264, grid=grid(43264), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 96, 13, 13), (16224, 1, 1248, 96))
        buf109 = empty_strided_cuda((4, 96, 13, 13), (16224, 1, 1248, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_68, branch3x3dbl_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf108, primals_143, primals_144, primals_145, primals_146, buf109, 64896, grid=grid(64896), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, buf15, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 96, 6, 6), (3456, 1, 576, 96))
        buf115 = empty_strided_cuda((4, 768, 6, 6), (27648, 36, 6, 1), torch.float32)
        buf111 = reinterpret_tensor(buf115, (4, 288, 6, 6), (27648, 36, 6, 1), 17280)  # alias
        buf112 = empty_strided_cuda((4, 288, 6, 6), (10368, 1, 1728, 288), torch.int8)
        # Topologically Sorted Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_33.run(buf104, buf111, buf112, 144, 288, grid=grid(144, 288), stream=stream0)
        buf113 = reinterpret_tensor(buf115, (4, 384, 6, 6), (27648, 36, 6, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_64, branch3x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf105, primals_133, primals_134, primals_135, primals_136, buf113, 1536, 36, grid=grid(1536, 36), stream=stream0)
        buf114 = reinterpret_tensor(buf115, (4, 96, 6, 6), (27648, 36, 6, 1), 13824)  # alias
        # Topologically Sorted Source Nodes: [x_70, branch3x3dbl_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf110, primals_148, primals_149, primals_150, primals_151, buf114, 384, 36, grid=grid(384, 36), stream=stream0)
        buf116 = empty_strided_cuda((4, 768, 6, 6), (27648, 1, 4608, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf115, buf116, 3072, 36, grid=grid(3072, 36), stream=stream0)
        del buf111
        del buf113
        del buf114
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 192, 6, 6), (6912, 1, 1152, 192))
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf116, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 128, 6, 6), (4608, 1, 768, 128))
        buf119 = empty_strided_cuda((4, 128, 6, 6), (4608, 1, 768, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_75, branch7x7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf118, primals_158, primals_159, primals_160, primals_161, buf119, 18432, grid=grid(18432), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, buf16, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 128, 6, 6), (4608, 1, 768, 128))
        buf121 = empty_strided_cuda((4, 128, 6, 6), (4608, 1, 768, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_77, branch7x7_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf120, primals_163, primals_164, primals_165, primals_166, buf121, 18432, grid=grid(18432), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, buf17, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 192, 6, 6), (6912, 1, 1152, 192))
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf116, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 128, 6, 6), (4608, 1, 768, 128))
        buf124 = empty_strided_cuda((4, 128, 6, 6), (4608, 1, 768, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_81, branch7x7dbl], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf123, primals_173, primals_174, primals_175, primals_176, buf124, 18432, grid=grid(18432), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, buf18, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 128, 6, 6), (4608, 1, 768, 128))
        buf126 = empty_strided_cuda((4, 128, 6, 6), (4608, 1, 768, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_83, branch7x7dbl_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf125, primals_178, primals_179, primals_180, primals_181, buf126, 18432, grid=grid(18432), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, buf19, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 128, 6, 6), (4608, 1, 768, 128))
        buf128 = empty_strided_cuda((4, 128, 6, 6), (4608, 1, 768, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_85, branch7x7dbl_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf127, primals_183, primals_184, primals_185, primals_186, buf128, 18432, grid=grid(18432), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, buf20, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 128, 6, 6), (4608, 1, 768, 128))
        buf130 = empty_strided_cuda((4, 128, 6, 6), (4608, 1, 768, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_87, branch7x7dbl_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf129, primals_188, primals_189, primals_190, primals_191, buf130, 18432, grid=grid(18432), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, buf21, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf132 = reinterpret_tensor(buf115, (4, 768, 6, 6), (27648, 1, 4608, 768), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [branch_pool_7], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf116, buf132, 110592, grid=grid(110592), stream=stream0)
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf134 = empty_strided_cuda((4, 768, 6, 6), (27648, 1, 4608, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_39.run(buf117, primals_153, primals_154, primals_155, primals_156, buf122, primals_168, primals_169, primals_170, primals_171, buf131, primals_193, primals_194, primals_195, primals_196, buf133, primals_198, primals_199, primals_200, primals_201, buf134, 110592, grid=grid(110592), stream=stream0)
        # Topologically Sorted Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 192, 6, 6), (6912, 1, 1152, 192))
        # Topologically Sorted Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf134, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf137 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_96, branch7x7_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf136, primals_208, primals_209, primals_210, primals_211, buf137, 23040, grid=grid(23040), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [x_97], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, buf22, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf139 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_98, branch7x7_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf138, primals_213, primals_214, primals_215, primals_216, buf139, 23040, grid=grid(23040), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, buf23, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 192, 6, 6), (6912, 1, 1152, 192))
        # Topologically Sorted Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf134, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf142 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_102, branch7x7dbl_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf141, primals_223, primals_224, primals_225, primals_226, buf142, 23040, grid=grid(23040), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [x_103], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, buf24, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf144 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_104, branch7x7dbl_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf143, primals_228, primals_229, primals_230, primals_231, buf144, 23040, grid=grid(23040), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, buf25, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf146 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_106, branch7x7dbl_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf145, primals_233, primals_234, primals_235, primals_236, buf146, 23040, grid=grid(23040), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, buf26, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf148 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_108, branch7x7dbl_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf147, primals_238, primals_239, primals_240, primals_241, buf148, 23040, grid=grid(23040), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, buf27, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf150 = empty_strided_cuda((4, 768, 6, 6), (27648, 1, 4608, 768), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_9], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf134, buf150, 110592, grid=grid(110592), stream=stream0)
        # Topologically Sorted Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf152 = empty_strided_cuda((4, 768, 6, 6), (27648, 1, 4608, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_113], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_39.run(buf135, primals_203, primals_204, primals_205, primals_206, buf140, primals_218, primals_219, primals_220, primals_221, buf149, primals_243, primals_244, primals_245, primals_246, buf151, primals_248, primals_249, primals_250, primals_251, buf152, 110592, grid=grid(110592), stream=stream0)
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 192, 6, 6), (6912, 1, 1152, 192))
        # Topologically Sorted Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf152, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf155 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_117, branch7x7_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf154, primals_258, primals_259, primals_260, primals_261, buf155, 23040, grid=grid(23040), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, buf28, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf157 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_119, branch7x7_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf156, primals_263, primals_264, primals_265, primals_266, buf157, 23040, grid=grid(23040), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, buf29, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 192, 6, 6), (6912, 1, 1152, 192))
        # Topologically Sorted Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf152, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf160 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_123, branch7x7dbl_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf159, primals_273, primals_274, primals_275, primals_276, buf160, 23040, grid=grid(23040), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, buf30, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf162 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_125, branch7x7dbl_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf161, primals_278, primals_279, primals_280, primals_281, buf162, 23040, grid=grid(23040), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, buf31, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf164 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_127, branch7x7dbl_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf163, primals_283, primals_284, primals_285, primals_286, buf164, 23040, grid=grid(23040), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, buf32, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 160, 6, 6), (5760, 1, 960, 160))
        buf166 = empty_strided_cuda((4, 160, 6, 6), (5760, 1, 960, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_129, branch7x7dbl_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf165, primals_288, primals_289, primals_290, primals_291, buf166, 23040, grid=grid(23040), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, buf33, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf168 = empty_strided_cuda((4, 768, 6, 6), (27648, 1, 4608, 768), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_11], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf152, buf168, 110592, grid=grid(110592), stream=stream0)
        # Topologically Sorted Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf170 = empty_strided_cuda((4, 768, 6, 6), (27648, 1, 4608, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_134], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_39.run(buf153, primals_253, primals_254, primals_255, primals_256, buf158, primals_268, primals_269, primals_270, primals_271, buf167, primals_293, primals_294, primals_295, primals_296, buf169, primals_298, primals_299, primals_300, primals_301, buf170, 110592, grid=grid(110592), stream=stream0)
        # Topologically Sorted Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 192, 6, 6), (6912, 1, 1152, 192))
        # Topologically Sorted Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf170, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf173 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_138, branch7x7_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf172, primals_308, primals_309, primals_310, primals_311, buf173, 27648, grid=grid(27648), stream=stream0)
        del primals_311
        # Topologically Sorted Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, buf34, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf175 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_140, branch7x7_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf174, primals_313, primals_314, primals_315, primals_316, buf175, 27648, grid=grid(27648), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, buf35, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 192, 6, 6), (6912, 1, 1152, 192))
        # Topologically Sorted Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf170, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf178 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_144, branch7x7dbl_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf177, primals_323, primals_324, primals_325, primals_326, buf178, 27648, grid=grid(27648), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [x_145], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, buf36, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf180 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_146, branch7x7dbl_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf179, primals_328, primals_329, primals_330, primals_331, buf180, 27648, grid=grid(27648), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [x_147], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, buf37, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf182 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_148, branch7x7dbl_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf181, primals_333, primals_334, primals_335, primals_336, buf182, 27648, grid=grid(27648), stream=stream0)
        del primals_336
        # Topologically Sorted Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, buf38, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf184 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_150, branch7x7dbl_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf183, primals_338, primals_339, primals_340, primals_341, buf184, 27648, grid=grid(27648), stream=stream0)
        del primals_341
        # Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, buf39, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf186 = empty_strided_cuda((4, 768, 6, 6), (27648, 1, 4608, 768), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_13], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf170, buf186, 110592, grid=grid(110592), stream=stream0)
        # Topologically Sorted Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf188 = empty_strided_cuda((4, 768, 6, 6), (27648, 1, 4608, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_155], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_39.run(buf171, primals_303, primals_304, primals_305, primals_306, buf176, primals_318, primals_319, primals_320, primals_321, buf185, primals_343, primals_344, primals_345, primals_346, buf187, primals_348, primals_349, primals_350, primals_351, buf188, 110592, grid=grid(110592), stream=stream0)
        # Topologically Sorted Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_352, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf190 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_157, branch3x3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf189, primals_353, primals_354, primals_355, primals_356, buf190, 27648, grid=grid(27648), stream=stream0)
        del primals_356
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, buf40, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 320, 2, 2), (1280, 1, 640, 320))
        # Topologically Sorted Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf188, primals_362, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf193 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_161, branch7x7x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf192, primals_363, primals_364, primals_365, primals_366, buf193, 27648, grid=grid(27648), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, buf41, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf195 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_163, branch7x7x3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf194, primals_368, primals_369, primals_370, primals_371, buf195, 27648, grid=grid(27648), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, buf42, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 192, 6, 6), (6912, 1, 1152, 192))
        buf197 = empty_strided_cuda((4, 192, 6, 6), (6912, 1, 1152, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_165, branch7x7x3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf196, primals_373, primals_374, primals_375, primals_376, buf197, 27648, grid=grid(27648), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, buf43, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 192, 2, 2), (768, 1, 384, 192))
        buf203 = empty_strided_cuda((4, 1280, 2, 2), (5120, 4, 2, 1), torch.float32)
        buf199 = reinterpret_tensor(buf203, (4, 768, 2, 2), (5120, 4, 2, 1), 2048)  # alias
        buf200 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.int8)
        # Topologically Sorted Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_42.run(buf188, buf199, buf200, 16, 768, grid=grid(16, 768), stream=stream0)
        buf201 = reinterpret_tensor(buf203, (4, 320, 2, 2), (5120, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_159, branch3x3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf191, primals_358, primals_359, primals_360, primals_361, buf201, 1280, 4, grid=grid(1280, 4), stream=stream0)
        buf202 = reinterpret_tensor(buf203, (4, 192, 2, 2), (5120, 4, 2, 1), 1280)  # alias
        # Topologically Sorted Source Nodes: [x_167, branch7x7x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf198, primals_378, primals_379, primals_380, primals_381, buf202, 768, 4, grid=grid(768, 4), stream=stream0)
        buf204 = empty_strided_cuda((4, 1280, 2, 2), (5120, 1, 2560, 1280), torch.float32)
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_45.run(buf203, buf204, 5120, 4, grid=grid(5120, 4), stream=stream0)
        del buf199
        del buf201
        del buf202
        # Topologically Sorted Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_382, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 320, 2, 2), (1280, 1, 640, 320))
        # Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf204, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf207 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_172, branch3x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf206, primals_388, primals_389, primals_390, primals_391, buf207, 6144, grid=grid(6144), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [x_173], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, buf44, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 384, 2, 2), (1536, 1, 768, 384))
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf207, buf45, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf222 = empty_strided_cuda((4, 2048, 2, 2), (8192, 4, 2, 1), torch.float32)
        buf210 = reinterpret_tensor(buf222, (4, 768, 2, 2), (8192, 4, 2, 1), 1280)  # alias
        # Topologically Sorted Source Nodes: [branch3x3_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_47.run(buf208, primals_393, primals_394, primals_395, primals_396, buf209, primals_398, primals_399, primals_400, primals_401, buf210, 12288, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf204, primals_402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 448, 2, 2), (1792, 1, 896, 448))
        buf212 = empty_strided_cuda((4, 448, 2, 2), (1792, 1, 896, 448), torch.float32)
        # Topologically Sorted Source Nodes: [x_178, branch3x3dbl_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf211, primals_403, primals_404, primals_405, primals_406, buf212, 7168, grid=grid(7168), stream=stream0)
        del primals_406
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf214 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_180, branch3x3dbl_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf213, primals_408, primals_409, primals_410, primals_411, buf214, 6144, grid=grid(6144), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, buf47, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 384, 2, 2), (1536, 1, 768, 384))
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf214, buf48, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf217 = reinterpret_tensor(buf222, (4, 768, 2, 2), (8192, 4, 2, 1), 4352)  # alias
        # Topologically Sorted Source Nodes: [branch3x3dbl_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_47.run(buf215, primals_413, primals_414, primals_415, primals_416, buf216, primals_418, primals_419, primals_420, primals_421, buf217, 12288, grid=grid(12288), stream=stream0)
        buf218 = reinterpret_tensor(buf203, (4, 1280, 2, 2), (5120, 1, 2560, 1280), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [branch_pool_16], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_49.run(buf204, buf218, 20480, grid=grid(20480), stream=stream0)
        # Topologically Sorted Source Nodes: [x_185], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 192, 2, 2), (768, 1, 384, 192))
        buf220 = reinterpret_tensor(buf222, (4, 320, 2, 2), (8192, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_170, branch1x1_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf205, primals_383, primals_384, primals_385, primals_386, buf220, 1280, 4, grid=grid(1280, 4), stream=stream0)
        buf221 = reinterpret_tensor(buf222, (4, 192, 2, 2), (8192, 4, 2, 1), 7424)  # alias
        # Topologically Sorted Source Nodes: [x_186, branch_pool_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf219, primals_423, primals_424, primals_425, primals_426, buf221, 768, 4, grid=grid(768, 4), stream=stream0)
        buf223 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_187], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_52.run(buf222, buf223, 8192, 4, grid=grid(8192, 4), stream=stream0)
        del buf210
        del buf217
        del buf220
        del buf221
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_427, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 320, 2, 2), (1280, 1, 640, 320))
        # Topologically Sorted Source Nodes: [x_190], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf223, primals_432, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf226 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_191, branch3x3_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf225, primals_433, primals_434, primals_435, primals_436, buf226, 6144, grid=grid(6144), stream=stream0)
        del primals_436
        # Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, buf49, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 384, 2, 2), (1536, 1, 768, 384))
        # Topologically Sorted Source Nodes: [x_194], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf50, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf241 = buf222; del buf222  # reuse
        buf229 = reinterpret_tensor(buf241, (4, 768, 2, 2), (8192, 4, 2, 1), 1280)  # alias
        # Topologically Sorted Source Nodes: [branch3x3_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_47.run(buf227, primals_438, primals_439, primals_440, primals_441, buf228, primals_443, primals_444, primals_445, primals_446, buf229, 12288, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf223, primals_447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 448, 2, 2), (1792, 1, 896, 448))
        buf231 = empty_strided_cuda((4, 448, 2, 2), (1792, 1, 896, 448), torch.float32)
        # Topologically Sorted Source Nodes: [x_197, branch3x3dbl_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf230, primals_448, primals_449, primals_450, primals_451, buf231, 7168, grid=grid(7168), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf233 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_199, branch3x3dbl_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf232, primals_453, primals_454, primals_455, primals_456, buf233, 6144, grid=grid(6144), stream=stream0)
        del primals_456
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, buf52, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 384, 2, 2), (1536, 1, 768, 384))
        # Topologically Sorted Source Nodes: [x_202], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf233, buf53, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf236 = reinterpret_tensor(buf241, (4, 768, 2, 2), (8192, 4, 2, 1), 4352)  # alias
        # Topologically Sorted Source Nodes: [branch3x3dbl_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_47.run(buf234, primals_458, primals_459, primals_460, primals_461, buf235, primals_463, primals_464, primals_465, primals_466, buf236, 12288, grid=grid(12288), stream=stream0)
        buf237 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_18], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_53.run(buf223, buf237, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_467, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 192, 2, 2), (768, 1, 384, 192))
        buf239 = reinterpret_tensor(buf241, (4, 320, 2, 2), (8192, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_189, branch1x1_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf224, primals_428, primals_429, primals_430, primals_431, buf239, 1280, 4, grid=grid(1280, 4), stream=stream0)
        buf240 = reinterpret_tensor(buf241, (4, 192, 2, 2), (8192, 4, 2, 1), 7424)  # alias
        # Topologically Sorted Source Nodes: [x_205, branch_pool_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf238, primals_468, primals_469, primals_470, primals_471, buf240, 768, 4, grid=grid(768, 4), stream=stream0)
        buf242 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_54.run(buf241, buf242, 8192, grid=grid(8192), stream=stream0)
        del buf229
        del buf236
        del buf239
        del buf240
        del buf241
        buf243 = empty_strided_cuda((4, 12), (12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_210], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_473, reinterpret_tensor(buf242, (4, 2048), (2048, 1), 0), reinterpret_tensor(primals_472, (2048, 12), (1, 2048), 0), alpha=1, beta=1, out=buf243)
        del primals_473
    return (buf243, buf0, primals_3, primals_4, primals_5, buf1, primals_8, primals_9, primals_10, buf2, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, buf3, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, buf4, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, buf5, primals_48, primals_49, primals_50, buf6, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, buf7, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, buf8, primals_83, primals_84, primals_85, buf9, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, buf10, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, buf11, primals_118, primals_119, primals_120, buf12, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, buf13, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, buf14, primals_143, primals_144, primals_145, buf15, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, buf16, primals_163, primals_164, primals_165, buf17, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, buf18, primals_178, primals_179, primals_180, buf19, primals_183, primals_184, primals_185, buf20, primals_188, primals_189, primals_190, buf21, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, buf22, primals_213, primals_214, primals_215, buf23, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, buf24, primals_228, primals_229, primals_230, buf25, primals_233, primals_234, primals_235, buf26, primals_238, primals_239, primals_240, buf27, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, buf28, primals_263, primals_264, primals_265, buf29, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, buf30, primals_278, primals_279, primals_280, buf31, primals_283, primals_284, primals_285, buf32, primals_288, primals_289, primals_290, buf33, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, buf34, primals_313, primals_314, primals_315, buf35, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, buf36, primals_328, primals_329, primals_330, buf37, primals_333, primals_334, primals_335, buf38, primals_338, primals_339, primals_340, buf39, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, buf40, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, buf41, primals_368, primals_369, primals_370, buf42, primals_373, primals_374, primals_375, buf43, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, buf44, primals_393, primals_394, primals_395, primals_396, buf45, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, buf46, primals_408, primals_409, primals_410, buf47, primals_413, primals_414, primals_415, primals_416, buf48, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, buf49, primals_438, primals_439, primals_440, primals_441, buf50, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, buf51, primals_453, primals_454, primals_455, buf52, primals_458, primals_459, primals_460, primals_461, buf53, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf112, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf128, buf129, buf130, buf131, buf132, buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf193, buf194, buf195, buf196, buf197, buf198, buf200, buf204, buf205, buf206, buf207, buf208, buf209, buf211, buf212, buf213, buf214, buf215, buf216, buf218, buf219, buf223, buf224, buf225, buf226, buf227, buf228, buf230, buf231, buf232, buf233, buf234, buf235, buf237, buf238, reinterpret_tensor(buf242, (4, 2048), (2048, 1), 0), primals_472, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 128, 128), (65536, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((80, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((384, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((448, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((192, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((320, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((384, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((448, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((12, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
