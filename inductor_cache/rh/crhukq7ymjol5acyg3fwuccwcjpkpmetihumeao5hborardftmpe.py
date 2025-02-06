# AOT ID: ['24_forward']
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


# kernel path: inductor_cache/oh/cohz4is6qajm5ak7h6bsarpfjh4aa6sdfnwhpc4d5cetiwwb3fok.py
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
    size_hints={'y': 16, 'x': 262144}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 262144
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
    tmp0 = tl.load(in_ptr0 + (x2 + 262144*y3), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 786432*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/yp/cyp7ls54zbitglsogi3ojjjwsiiwvukg4xmuq4262izpl2gllpco.py
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


# kernel path: inductor_cache/32/c32bd6kos5g3xs2rpdo4j24uxjn3wt2s4nhwmfhn45zp4czqwuq2.py
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
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/wy/cwy3y67vevgajhebutpwnzc23fg4bqb6xjkoru3nqbbh7lpaekww.py
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


# kernel path: inductor_cache/7e/c7eupdhxzoyka4rtoqjrbopx677vdqmq5upzfecyo2dcbnj6tnly.py
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
    size_hints={'y': 4096, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/yx/cyx6rkqgq7qu7zvhhhnmj744xh37e2mnztm3fiatsisjcfielmv4.py
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


# kernel path: inductor_cache/ah/cah3varn2ppjmbk225y5tr5334jdoydlcnqu32n23g6mld4thnzk.py
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


# kernel path: inductor_cache/kf/ckfyz4ybnknd53kzymgo4ni7j4zkz7oyg76ygth56rrajx24unso.py
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
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/2s/c2s4nincqecedrafapdnagg42efd2pytxynadvuebmo52a4nd5lu.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 48*x2 + 432*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oh/cohofwbfu57ott4fxkaijnvss5d2wprut7khexopl6nathmxf3zv.py
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
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 122880
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 320*x2 + 2880*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ip/cipa2dlcdwxplqj4htpqmjl3acrctsriqhn62m52zbbr2xzcwzcf.py
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/cg/ccgynue4zuko2bmz5q4dvfolcmhdmjcnfgou7zrug7rhn6jhwnuw.py
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
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
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


# kernel path: inductor_cache/kg/ckg5jyz6voqyuwx7hslq7h4gc4g3ueego5f46gtogttnqbb5hkej.py
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
    size_hints={'y': 32768, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
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


# kernel path: inductor_cache/lj/cljl6e3xdyqiexgo7th4mhblvi3unyistcwm6g7kyfewdx4xuzr5.py
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
    size_hints={'y': 32768, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/pc/cpcfjcvdzdhvq2havkq7iqzjr53sqiqgza4toi5wnfdouyoxtep2.py
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
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ai/caicubnhwva7bizzkzmqfulq55uwhqwnrndodwsx4nq43qpqjlhe.py
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
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 92160
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


# kernel path: inductor_cache/wg/cwgxkj236emwlhhn6t7oddopim2itodqwd4mac35na3hcpaief74.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_17 = async_compile.triton('triton_poi_fused_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 43008
    xnumel = 3
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
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mv/cmvzcyzuduwk7eybb5ldi63bp5746ic6x6cepx5hpu5qtwopt3bc.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_18 = async_compile.triton('triton_poi_fused_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 57344
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 224)
    y1 = yindex // 224
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 224*x2 + 672*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cp/ccpee3uvjjadakljuyrilecohtisn7bo25mnxzgw6xbf66izen6k.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8323200
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


# kernel path: inductor_cache/rw/crwibt5ehx7lpbm7nk4umsqnhdqz4hfwvj37qhnloablbh2f3mu6.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_4 => add_3, mul_4, mul_5, sub_1
#   x_5 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8193152
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


# kernel path: inductor_cache/cj/ccj5f2zsuh5dbcl7r22niwvn3riqn7h56cdskr37bf4krtgvze76.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_7 => add_5, mul_7, mul_8, sub_2
#   x_8 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16386304
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


# kernel path: inductor_cache/fg/cfgrp3sdf6priphjhrokolivd6dqttp3wz5oalzl264bf44bfuro.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_9 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_22 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_22(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4064256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 126)
    x2 = ((xindex // 8064) % 126)
    x3 = xindex // 1016064
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 32384*x2 + 4096576*x3), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 32384*x2 + 4096576*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 128*x1 + 32384*x2 + 4096576*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (16192 + x0 + 128*x1 + 32384*x2 + 4096576*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (16256 + x0 + 128*x1 + 32384*x2 + 4096576*x3), xmask)
    tmp9 = tl.load(in_ptr0 + (16320 + x0 + 128*x1 + 32384*x2 + 4096576*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (32384 + x0 + 128*x1 + 32384*x2 + 4096576*x3), xmask)
    tmp13 = tl.load(in_ptr0 + (32448 + x0 + 128*x1 + 32384*x2 + 4096576*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (32512 + x0 + 128*x1 + 32384*x2 + 4096576*x3), xmask)
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


# kernel path: inductor_cache/rc/crck27eukgudsrswcem3azlg52ldoexabwll566cb4c47nirucgt.py
# Topologically Sorted Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_11 => add_7, mul_10, mul_11, sub_3
#   x_12 => relu_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5080320
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


# kernel path: inductor_cache/ru/cruphwgd7se367vg6oiasf465jjbafgpxezicbamaaqonn444rsg.py
# Topologically Sorted Source Nodes: [x_14, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_14 => add_9, mul_13, mul_14, sub_4
#   x_15 => relu_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11808768
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


# kernel path: inductor_cache/n3/cn3eflrckozq4b6u3fvixtnwyxf6l73ziw3s3oq6qj6pbylr56n6.py
# Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_16 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=5] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_25 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_25(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2857728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 61)
    x2 = ((xindex // 11712) % 61)
    x3 = xindex // 714432
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1 + 47616*x2 + 2952192*x3), xmask)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + 384*x1 + 47616*x2 + 2952192*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (384 + x0 + 384*x1 + 47616*x2 + 2952192*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (23808 + x0 + 384*x1 + 47616*x2 + 2952192*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (24000 + x0 + 384*x1 + 47616*x2 + 2952192*x3), xmask)
    tmp9 = tl.load(in_ptr0 + (24192 + x0 + 384*x1 + 47616*x2 + 2952192*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (47616 + x0 + 384*x1 + 47616*x2 + 2952192*x3), xmask)
    tmp13 = tl.load(in_ptr0 + (47808 + x0 + 384*x1 + 47616*x2 + 2952192*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (48000 + x0 + 384*x1 + 47616*x2 + 2952192*x3), xmask)
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


# kernel path: inductor_cache/am/cam3m7azjctwyhyptiw77jg2wqz4wrc45b7cdnddqzge7hvo3jsb.py
# Topologically Sorted Source Nodes: [x_21, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_21 => add_13, mul_19, mul_20, sub_6
#   x_22 => relu_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 714432
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


# kernel path: inductor_cache/hb/chbizcq7zfxasaj36ju2ovttqqyybirtplrzvfn7bgpfvcnb4hi2.py
# Topologically Sorted Source Nodes: [x_27, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_27 => add_17, mul_25, mul_26, sub_8
#   x_28 => relu_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/6k/c6kuc6ngtkfco2k7znh7gxscfnoqxbb7cqziifda3qyxlvhmjuub.py
# Topologically Sorted Source Nodes: [x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_30 => add_19, mul_28, mul_29, sub_9
#   x_31 => relu_9
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1428864
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


# kernel path: inductor_cache/ms/cmskqa5y6sqyajycfya5qgycmj4m4hmuy7a34ck6k2ppszkwpw6p.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_2, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_29 = async_compile.triton('triton_poi_fused_avg_pool2d_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2857728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 11712) % 61)
    x1 = ((xindex // 192) % 61)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 61, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-11904) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-11712) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-11520) + x6), tmp23 & xmask, other=0.0)
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
    tmp44 = tl.load(in_ptr0 + (11520 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (11712 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (11904 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0))) + ((61) * ((61) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (61)))*((61) * ((61) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (61))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((61) * ((61) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (61)))) + ((-1)*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))*((61) * ((61) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (61))))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jp/cjppnqneafo7omn7gnuttww6las3nb7uszd7arr6nn26pucqy3pq.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out => cat
# Graph fragment:
#   %cat : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_5, %relu_7, %relu_10, %relu_11], 1), kwargs = {})
triton_poi_fused_cat_30 = async_compile.triton('triton_poi_fused_cat_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4762880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 320)
    x1 = xindex // 320
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (96*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp26 = tl.full([1], 160, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (64*x1 + ((-96) + x0)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-96) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-96) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-96) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-96) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 256, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (96*x1 + ((-160) + x0)), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-160) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-160) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = 1e-05
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-160) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-160) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 320, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (64*x1 + ((-256) + x0)), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-256) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-256) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = 1e-05
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-256) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-256) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: inductor_cache/5f/c5f5b7sttrbyiyu6ryvyeod7w7ryvmofznwuf5ux32gbxf4ao34h.py
# Topologically Sorted Source Nodes: [x_42, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_42 => add_27, mul_40, mul_41, sub_13
#   x_43 => relu_13
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_27,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/2l/c2lzh42advxgd44kh7mk7qqq236q2gjiiqn2qaugehqkf5kl4v3e.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %relu_14, %relu_17], 1), kwargs = {})
triton_poi_fused_cat_32 = async_compile.triton('triton_poi_fused_cat_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1905152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp26 = tl.full([1], 64, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (32*x1 + ((-32) + x0)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-32) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-32) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-32) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-32) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 128, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tl.load(in_ptr10 + (64*x1 + ((-64) + x0)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = tl.load(in_ptr11 + ((-64) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tmp52 - tmp53
    tmp55 = tl.load(in_ptr12 + ((-64) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = 1e-05
    tmp57 = tmp55 + tmp56
    tmp58 = libdevice.sqrt(tmp57)
    tmp59 = tl.full([1], 1, tl.int32)
    tmp60 = tmp59 / tmp58
    tmp61 = 1.0
    tmp62 = tmp60 * tmp61
    tmp63 = tmp54 * tmp62
    tmp64 = tl.load(in_ptr13 + ((-64) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr14 + ((-64) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.full([1], 0, tl.int32)
    tmp69 = triton_helpers.maximum(tmp68, tmp67)
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp49, tmp69, tmp70)
    tmp72 = tl.where(tmp28, tmp48, tmp71)
    tmp73 = tl.where(tmp4, tmp24, tmp72)
    tl.store(out_ptr0 + (x2), tmp73, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/es/ceslar2h4sen4byc437wlcwogtczypmdau6lysst5ajfkw7anxqn.py
# Topologically Sorted Source Nodes: [out_2, mul, out_3, out_4], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
# Source node to ATen node mapping:
#   mul => mul_54
#   out_2 => convolution_18
#   out_3 => add_36
#   out_4 => relu_18
# Graph fragment:
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %primals_92, %primals_93, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_18, 0.17), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, %cat), kwargs = {})
#   %relu_18 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_36,), kwargs = {})
triton_poi_fused_add_convolution_mul_relu_33 = async_compile.triton('triton_poi_fused_add_convolution_mul_relu_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_relu_33(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4762880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.17
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ra/craxazxcszekgsjm4wl2aivzrqllemdpwoxa4s5aegjqt3m73dak.py
# Topologically Sorted Source Nodes: [x_222, x_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_222 => add_157, mul_230, mul_231, sub_73
#   x_223 => relu_83
# Graph fragment:
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_585), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_587), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_230, %unsqueeze_589), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_231, %unsqueeze_591), kwargs = {})
#   %relu_83 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_157,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3810304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
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


# kernel path: inductor_cache/ue/cue5kog43wt2cm6pln5t5jlidc5d7g4yiffbaj6g4ko3xjhcmct5.py
# Topologically Sorted Source Nodes: [x2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x2 => _low_memory_max_pool2d_with_offsets_2, getitem_5
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_81, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_35 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_35(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3600
    xnumel = 320
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 30)
    y1 = ((yindex // 30) % 30)
    y2 = yindex // 900
    y4 = (yindex % 900)
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + 640*y0 + 39040*y1 + 1190720*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (320 + x3 + 640*y0 + 39040*y1 + 1190720*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (640 + x3 + 640*y0 + 39040*y1 + 1190720*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (19520 + x3 + 640*y0 + 39040*y1 + 1190720*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (19840 + x3 + 640*y0 + 39040*y1 + 1190720*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (20160 + x3 + 640*y0 + 39040*y1 + 1190720*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (39040 + x3 + 640*y0 + 39040*y1 + 1190720*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (39360 + x3 + 640*y0 + 39040*y1 + 1190720*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (39680 + x3 + 640*y0 + 39040*y1 + 1190720*y2), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y4 + 900*x3 + 979200*y2), tmp16, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 320*y5), tmp41, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cikhr7bldcreub6o3l5mizqzyza6oz7mrc3tgws44bakryya6ks2.py
# Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_219 => add_155, mul_227, mul_228, sub_72
#   x_220 => relu_82
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_82, %unsqueeze_577), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_227, %unsqueeze_581), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_228, %unsqueeze_583), kwargs = {})
#   %relu_82 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 900
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (y0 + 384*x2 + 345600*y1), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 900*y0 + 979200*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/aw/cawttq6c5w4lrtgpjqfbkvvil5mznaqmsa6qpjc4erxsxwmm2ofh.py
# Topologically Sorted Source Nodes: [out_41], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_41 => cat_11
# Graph fragment:
#   %cat_11 : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_82, %relu_85, %getitem_4], 1), kwargs = {})
triton_poi_fused_cat_37 = async_compile.triton('triton_poi_fused_cat_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_37(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4352
    xnumel = 900
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1088)
    y1 = yindex // 1088
    tmp0 = tl.load(in_ptr0 + (x2 + 900*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1088*x2 + 979200*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/2n/c2n4jg4ll6jtkzk2hvchguej5x3its3xk3w7pdblwvmqlevmtqwb.py
# Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_234 => add_165, mul_242, mul_243, sub_77
#   x_235 => relu_87
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_617), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_242, %unsqueeze_621), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_243, %unsqueeze_623), kwargs = {})
#   %relu_87 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_165,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 460800
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


# kernel path: inductor_cache/kl/ckloob2ublh7ooehdmlza7hwlwnc4gwgyl3vop3e7uvlbc4nq6dy.py
# Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_237 => add_167, mul_245, mul_246, sub_78
#   x_238 => relu_88
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_625), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_245, %unsqueeze_629), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_246, %unsqueeze_631), kwargs = {})
#   %relu_88 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_167,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576000
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


# kernel path: inductor_cache/ao/caojxoiiliitmopnhhfwmu4kb26nqaecd6uyizr56immmhsc6xky.py
# Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_42 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_86, %relu_89], 1), kwargs = {})
triton_poi_fused_cat_40 = async_compile.triton('triton_poi_fused_cat_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1382400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = xindex // 384
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (192*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp26 = tl.full([1], 384, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (192*x1 + ((-192) + x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-192) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-192) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-192) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-192) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x2), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rn/crnlbut7k27d3vnkeq6azo2ixygqdlcdlk4h3psp3sy5y3u2552l.py
# Topologically Sorted Source Nodes: [out_43, mul_10, out_44, out_45], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
# Source node to ATen node mapping:
#   mul_10 => mul_250
#   out_43 => convolution_90
#   out_44 => add_170
#   out_45 => relu_90
# Graph fragment:
#   %convolution_90 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_12, %primals_422, %primals_423, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_90, 0.1), kwargs = {})
#   %add_170 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_250, %cat_11), kwargs = {})
#   %relu_90 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_170,), kwargs = {})
triton_poi_fused_add_convolution_mul_relu_41 = async_compile.triton('triton_poi_fused_add_convolution_mul_relu_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_relu_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_relu_41(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3916800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1088)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vp/cvpkx6bluzlreqfib6qmpgdjrflairwzccdamnqmdjel3e5jyttt.py
# Topologically Sorted Source Nodes: [x_471, x_472], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_471 => add_343, mul_499, mul_500, sub_156
#   x_472 => relu_186
# Graph fragment:
#   %sub_156 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_186, %unsqueeze_1249), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_156, %unsqueeze_1251), kwargs = {})
#   %mul_500 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_499, %unsqueeze_1253), kwargs = {})
#   %add_343 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_500, %unsqueeze_1255), kwargs = {})
#   %relu_186 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_343,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 921600
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/oi/coic3zphplrqzcrhbpxlhegf3k3pbjutaltyrozrixwd2kmbtlfy.py
# Topologically Sorted Source Nodes: [x_486, x_487], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_486 => add_353, mul_514, mul_515, sub_161
#   x_487 => relu_191
# Graph fragment:
#   %sub_161 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_191, %unsqueeze_1289), kwargs = {})
#   %mul_514 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_161, %unsqueeze_1291), kwargs = {})
#   %mul_515 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_514, %unsqueeze_1293), kwargs = {})
#   %add_353 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_515, %unsqueeze_1295), kwargs = {})
#   %relu_191 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_353,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1036800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 288)
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


# kernel path: inductor_cache/6n/c6n577oubqu2fwpw6lvksab23xckfrh33demqduh67vujdt4vafa.py
# Topologically Sorted Source Nodes: [x3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x3 => _low_memory_max_pool2d_with_offsets_3, getitem_7
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_185, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_44 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_44(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 1088
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 14)
    y1 = ((yindex // 14) % 14)
    y2 = yindex // 196
    y4 = (yindex % 196)
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + 2176*y0 + 65280*y1 + 979200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1088 + x3 + 2176*y0 + 65280*y1 + 979200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2176 + x3 + 2176*y0 + 65280*y1 + 979200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (32640 + x3 + 2176*y0 + 65280*y1 + 979200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (33728 + x3 + 2176*y0 + 65280*y1 + 979200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (34816 + x3 + 2176*y0 + 65280*y1 + 979200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (65280 + x3 + 2176*y0 + 65280*y1 + 979200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (66368 + x3 + 2176*y0 + 65280*y1 + 979200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (67456 + x3 + 2176*y0 + 65280*y1 + 979200*y2), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y4 + 196*x3 + 407680*y2), tmp16, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 1088*y5), tmp41, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ol/colfn2qo6eov3g24a2wz626oyea7jvykgngflx7dybl3outlioia.py
# Topologically Sorted Source Nodes: [x_474, x_475], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_474 => add_345, mul_502, mul_503, sub_157
#   x_475 => relu_187
# Graph fragment:
#   %sub_157 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_187, %unsqueeze_1257), kwargs = {})
#   %mul_502 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_157, %unsqueeze_1259), kwargs = {})
#   %mul_503 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_502, %unsqueeze_1261), kwargs = {})
#   %add_345 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_503, %unsqueeze_1263), kwargs = {})
#   %relu_187 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_345,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (y0 + 384*x2 + 75264*y1), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 196*y0 + 407680*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/3z/c3zzom4ves7hp455qjr2yc622vl73h5xm6ud4cimftjqwamjwqme.py
# Topologically Sorted Source Nodes: [x_480, x_481], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_480 => add_349, mul_508, mul_509, sub_159
#   x_481 => relu_189
# Graph fragment:
#   %sub_159 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_189, %unsqueeze_1273), kwargs = {})
#   %mul_508 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_159, %unsqueeze_1275), kwargs = {})
#   %mul_509 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_508, %unsqueeze_1277), kwargs = {})
#   %add_349 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_509, %unsqueeze_1279), kwargs = {})
#   %relu_189 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_349,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 288)
    y1 = yindex // 288
    tmp0 = tl.load(in_ptr0 + (y0 + 288*x2 + 56448*y1), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 196*y0 + 407680*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/gh/cghvm3zmrcglhkbgxzmgylf3syd6caghhmyo4sd4sxbvdr2ifnav.py
# Topologically Sorted Source Nodes: [x_489, x_490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_489 => add_355, mul_517, mul_518, sub_162
#   x_490 => relu_192
# Graph fragment:
#   %sub_162 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_192, %unsqueeze_1297), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_162, %unsqueeze_1299), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_517, %unsqueeze_1301), kwargs = {})
#   %add_355 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_518, %unsqueeze_1303), kwargs = {})
#   %relu_192 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_355,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (y0 + 320*x2 + 62720*y1), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 196*y0 + 407680*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/gs/cgseme32n42vg3zv4smbvd3bo64z5rhkmbgpomj6i4lp7eyccy5z.py
# Topologically Sorted Source Nodes: [out_122], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_122 => cat_32
# Graph fragment:
#   %cat_32 : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_187, %relu_189, %relu_192, %getitem_6], 1), kwargs = {})
triton_poi_fused_cat_48 = async_compile.triton('triton_poi_fused_cat_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_48(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8320
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2080)
    y1 = yindex // 2080
    tmp0 = tl.load(in_ptr0 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 2080*x2 + 407680*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/7j/c7ja44gsfbkb6dan7fwvw7gbhh7cw32d3znyi6loug5h6ewmz464.py
# Topologically Sorted Source Nodes: [x_495, x_496], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_495 => add_359, mul_523, mul_524, sub_164
#   x_496 => relu_194
# Graph fragment:
#   %sub_164 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_194, %unsqueeze_1313), kwargs = {})
#   %mul_523 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_164, %unsqueeze_1315), kwargs = {})
#   %mul_524 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_523, %unsqueeze_1317), kwargs = {})
#   %add_359 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_524, %unsqueeze_1319), kwargs = {})
#   %relu_194 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_359,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
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


# kernel path: inductor_cache/bz/cbzvna7e5o3lq35fxa5qjrvsiylvs4ffwq53323gwde4v6twa6tr.py
# Topologically Sorted Source Nodes: [x_498, x_499], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_498 => add_361, mul_526, mul_527, sub_165
#   x_499 => relu_195
# Graph fragment:
#   %sub_165 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_195, %unsqueeze_1321), kwargs = {})
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_165, %unsqueeze_1323), kwargs = {})
#   %mul_527 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_526, %unsqueeze_1325), kwargs = {})
#   %add_361 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_527, %unsqueeze_1327), kwargs = {})
#   %relu_195 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_361,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 224)
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


# kernel path: inductor_cache/l7/cl7xfgzjxznpb2vvwbbtrrxwkijxe5ydbqh7q56lywfa2wwcmbur.py
# Topologically Sorted Source Nodes: [out_123], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_123 => cat_33
# Graph fragment:
#   %cat_33 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_193, %relu_196], 1), kwargs = {})
triton_poi_fused_cat_51 = async_compile.triton('triton_poi_fused_cat_51', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 448)
    x1 = xindex // 448
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (192*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp26 = tl.full([1], 448, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (256*x1 + ((-192) + x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-192) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-192) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-192) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-192) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x2), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dw/cdwkklfbdcnquclswrwciihgcyh5ghl3hzaey2aqrugveiij4umz.py
# Topologically Sorted Source Nodes: [out_124, mul_30, out_125, out_126], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
# Source node to ATen node mapping:
#   mul_30 => mul_531
#   out_124 => convolution_197
#   out_125 => add_364
#   out_126 => relu_197
# Graph fragment:
#   %convolution_197 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_33, %primals_897, %primals_898, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_531 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_197, 0.2), kwargs = {})
#   %add_364 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_531, %cat_32), kwargs = {})
#   %relu_197 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_364,), kwargs = {})
triton_poi_fused_add_convolution_mul_relu_52 = async_compile.triton('triton_poi_fused_add_convolution_mul_relu_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_relu_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_relu_52(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1630720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 2080)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5y/c5yrn75uyuh6pcgjq2snyxyy7vwuggizthqp23bn3layfuosangc.py
# Topologically Sorted Source Nodes: [out_160, mul_39, out_161], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_39 => mul_648
#   out_160 => convolution_242
#   out_161 => add_445
# Graph fragment:
#   %convolution_242 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_42, %primals_1095, %primals_1096, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_648 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_242, 1.0), kwargs = {})
#   %add_445 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_648, %relu_237), kwargs = {})
triton_poi_fused_add_convolution_mul_53 = async_compile.triton('triton_poi_fused_add_convolution_mul_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_53(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1630720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 2080)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ph/cphtgjogmfza6idjunlcwzmhnby2lip6fhed6lgdoiqaddavv5hi.py
# Topologically Sorted Source Nodes: [x_612, x_613], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_612 => add_447, mul_650, mul_651, sub_203
#   x_613 => relu_242
# Graph fragment:
#   %sub_203 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_243, %unsqueeze_1625), kwargs = {})
#   %mul_650 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_203, %unsqueeze_1627), kwargs = {})
#   %mul_651 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_650, %unsqueeze_1629), kwargs = {})
#   %add_447 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_651, %unsqueeze_1631), kwargs = {})
#   %relu_242 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_447,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_54 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_54', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1536)
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 512, 512), (786432, 262144, 512, 1))
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
    assert_size_stride(primals_27, (96, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_28, (96, ), (1, ))
    assert_size_stride(primals_29, (96, ), (1, ))
    assert_size_stride(primals_30, (96, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
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
    assert_size_stride(primals_57, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_58, (64, ), (1, ))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (64, ), (1, ))
    assert_size_stride(primals_62, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_63, (32, ), (1, ))
    assert_size_stride(primals_64, (32, ), (1, ))
    assert_size_stride(primals_65, (32, ), (1, ))
    assert_size_stride(primals_66, (32, ), (1, ))
    assert_size_stride(primals_67, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_68, (32, ), (1, ))
    assert_size_stride(primals_69, (32, ), (1, ))
    assert_size_stride(primals_70, (32, ), (1, ))
    assert_size_stride(primals_71, (32, ), (1, ))
    assert_size_stride(primals_72, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_73, (32, ), (1, ))
    assert_size_stride(primals_74, (32, ), (1, ))
    assert_size_stride(primals_75, (32, ), (1, ))
    assert_size_stride(primals_76, (32, ), (1, ))
    assert_size_stride(primals_77, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_78, (32, ), (1, ))
    assert_size_stride(primals_79, (32, ), (1, ))
    assert_size_stride(primals_80, (32, ), (1, ))
    assert_size_stride(primals_81, (32, ), (1, ))
    assert_size_stride(primals_82, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_83, (48, ), (1, ))
    assert_size_stride(primals_84, (48, ), (1, ))
    assert_size_stride(primals_85, (48, ), (1, ))
    assert_size_stride(primals_86, (48, ), (1, ))
    assert_size_stride(primals_87, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_88, (64, ), (1, ))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_92, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_93, (320, ), (1, ))
    assert_size_stride(primals_94, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, ), (1, ))
    assert_size_stride(primals_97, (32, ), (1, ))
    assert_size_stride(primals_98, (32, ), (1, ))
    assert_size_stride(primals_99, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_100, (32, ), (1, ))
    assert_size_stride(primals_101, (32, ), (1, ))
    assert_size_stride(primals_102, (32, ), (1, ))
    assert_size_stride(primals_103, (32, ), (1, ))
    assert_size_stride(primals_104, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_105, (32, ), (1, ))
    assert_size_stride(primals_106, (32, ), (1, ))
    assert_size_stride(primals_107, (32, ), (1, ))
    assert_size_stride(primals_108, (32, ), (1, ))
    assert_size_stride(primals_109, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_110, (32, ), (1, ))
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (32, ), (1, ))
    assert_size_stride(primals_113, (32, ), (1, ))
    assert_size_stride(primals_114, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_115, (48, ), (1, ))
    assert_size_stride(primals_116, (48, ), (1, ))
    assert_size_stride(primals_117, (48, ), (1, ))
    assert_size_stride(primals_118, (48, ), (1, ))
    assert_size_stride(primals_119, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (64, ), (1, ))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_125, (320, ), (1, ))
    assert_size_stride(primals_126, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_127, (32, ), (1, ))
    assert_size_stride(primals_128, (32, ), (1, ))
    assert_size_stride(primals_129, (32, ), (1, ))
    assert_size_stride(primals_130, (32, ), (1, ))
    assert_size_stride(primals_131, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_132, (32, ), (1, ))
    assert_size_stride(primals_133, (32, ), (1, ))
    assert_size_stride(primals_134, (32, ), (1, ))
    assert_size_stride(primals_135, (32, ), (1, ))
    assert_size_stride(primals_136, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_137, (32, ), (1, ))
    assert_size_stride(primals_138, (32, ), (1, ))
    assert_size_stride(primals_139, (32, ), (1, ))
    assert_size_stride(primals_140, (32, ), (1, ))
    assert_size_stride(primals_141, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_142, (32, ), (1, ))
    assert_size_stride(primals_143, (32, ), (1, ))
    assert_size_stride(primals_144, (32, ), (1, ))
    assert_size_stride(primals_145, (32, ), (1, ))
    assert_size_stride(primals_146, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_147, (48, ), (1, ))
    assert_size_stride(primals_148, (48, ), (1, ))
    assert_size_stride(primals_149, (48, ), (1, ))
    assert_size_stride(primals_150, (48, ), (1, ))
    assert_size_stride(primals_151, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_152, (64, ), (1, ))
    assert_size_stride(primals_153, (64, ), (1, ))
    assert_size_stride(primals_154, (64, ), (1, ))
    assert_size_stride(primals_155, (64, ), (1, ))
    assert_size_stride(primals_156, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_157, (320, ), (1, ))
    assert_size_stride(primals_158, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_161, (32, ), (1, ))
    assert_size_stride(primals_162, (32, ), (1, ))
    assert_size_stride(primals_163, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_164, (32, ), (1, ))
    assert_size_stride(primals_165, (32, ), (1, ))
    assert_size_stride(primals_166, (32, ), (1, ))
    assert_size_stride(primals_167, (32, ), (1, ))
    assert_size_stride(primals_168, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_169, (32, ), (1, ))
    assert_size_stride(primals_170, (32, ), (1, ))
    assert_size_stride(primals_171, (32, ), (1, ))
    assert_size_stride(primals_172, (32, ), (1, ))
    assert_size_stride(primals_173, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_174, (32, ), (1, ))
    assert_size_stride(primals_175, (32, ), (1, ))
    assert_size_stride(primals_176, (32, ), (1, ))
    assert_size_stride(primals_177, (32, ), (1, ))
    assert_size_stride(primals_178, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_179, (48, ), (1, ))
    assert_size_stride(primals_180, (48, ), (1, ))
    assert_size_stride(primals_181, (48, ), (1, ))
    assert_size_stride(primals_182, (48, ), (1, ))
    assert_size_stride(primals_183, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_189, (320, ), (1, ))
    assert_size_stride(primals_190, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_191, (32, ), (1, ))
    assert_size_stride(primals_192, (32, ), (1, ))
    assert_size_stride(primals_193, (32, ), (1, ))
    assert_size_stride(primals_194, (32, ), (1, ))
    assert_size_stride(primals_195, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_196, (32, ), (1, ))
    assert_size_stride(primals_197, (32, ), (1, ))
    assert_size_stride(primals_198, (32, ), (1, ))
    assert_size_stride(primals_199, (32, ), (1, ))
    assert_size_stride(primals_200, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (32, ), (1, ))
    assert_size_stride(primals_203, (32, ), (1, ))
    assert_size_stride(primals_204, (32, ), (1, ))
    assert_size_stride(primals_205, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_206, (32, ), (1, ))
    assert_size_stride(primals_207, (32, ), (1, ))
    assert_size_stride(primals_208, (32, ), (1, ))
    assert_size_stride(primals_209, (32, ), (1, ))
    assert_size_stride(primals_210, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_211, (48, ), (1, ))
    assert_size_stride(primals_212, (48, ), (1, ))
    assert_size_stride(primals_213, (48, ), (1, ))
    assert_size_stride(primals_214, (48, ), (1, ))
    assert_size_stride(primals_215, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_216, (64, ), (1, ))
    assert_size_stride(primals_217, (64, ), (1, ))
    assert_size_stride(primals_218, (64, ), (1, ))
    assert_size_stride(primals_219, (64, ), (1, ))
    assert_size_stride(primals_220, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_221, (320, ), (1, ))
    assert_size_stride(primals_222, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_223, (32, ), (1, ))
    assert_size_stride(primals_224, (32, ), (1, ))
    assert_size_stride(primals_225, (32, ), (1, ))
    assert_size_stride(primals_226, (32, ), (1, ))
    assert_size_stride(primals_227, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_228, (32, ), (1, ))
    assert_size_stride(primals_229, (32, ), (1, ))
    assert_size_stride(primals_230, (32, ), (1, ))
    assert_size_stride(primals_231, (32, ), (1, ))
    assert_size_stride(primals_232, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_233, (32, ), (1, ))
    assert_size_stride(primals_234, (32, ), (1, ))
    assert_size_stride(primals_235, (32, ), (1, ))
    assert_size_stride(primals_236, (32, ), (1, ))
    assert_size_stride(primals_237, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_238, (32, ), (1, ))
    assert_size_stride(primals_239, (32, ), (1, ))
    assert_size_stride(primals_240, (32, ), (1, ))
    assert_size_stride(primals_241, (32, ), (1, ))
    assert_size_stride(primals_242, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_243, (48, ), (1, ))
    assert_size_stride(primals_244, (48, ), (1, ))
    assert_size_stride(primals_245, (48, ), (1, ))
    assert_size_stride(primals_246, (48, ), (1, ))
    assert_size_stride(primals_247, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_248, (64, ), (1, ))
    assert_size_stride(primals_249, (64, ), (1, ))
    assert_size_stride(primals_250, (64, ), (1, ))
    assert_size_stride(primals_251, (64, ), (1, ))
    assert_size_stride(primals_252, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_253, (320, ), (1, ))
    assert_size_stride(primals_254, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_255, (32, ), (1, ))
    assert_size_stride(primals_256, (32, ), (1, ))
    assert_size_stride(primals_257, (32, ), (1, ))
    assert_size_stride(primals_258, (32, ), (1, ))
    assert_size_stride(primals_259, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_260, (32, ), (1, ))
    assert_size_stride(primals_261, (32, ), (1, ))
    assert_size_stride(primals_262, (32, ), (1, ))
    assert_size_stride(primals_263, (32, ), (1, ))
    assert_size_stride(primals_264, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, ), (1, ))
    assert_size_stride(primals_267, (32, ), (1, ))
    assert_size_stride(primals_268, (32, ), (1, ))
    assert_size_stride(primals_269, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_270, (32, ), (1, ))
    assert_size_stride(primals_271, (32, ), (1, ))
    assert_size_stride(primals_272, (32, ), (1, ))
    assert_size_stride(primals_273, (32, ), (1, ))
    assert_size_stride(primals_274, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_275, (48, ), (1, ))
    assert_size_stride(primals_276, (48, ), (1, ))
    assert_size_stride(primals_277, (48, ), (1, ))
    assert_size_stride(primals_278, (48, ), (1, ))
    assert_size_stride(primals_279, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_280, (64, ), (1, ))
    assert_size_stride(primals_281, (64, ), (1, ))
    assert_size_stride(primals_282, (64, ), (1, ))
    assert_size_stride(primals_283, (64, ), (1, ))
    assert_size_stride(primals_284, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_285, (320, ), (1, ))
    assert_size_stride(primals_286, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_287, (32, ), (1, ))
    assert_size_stride(primals_288, (32, ), (1, ))
    assert_size_stride(primals_289, (32, ), (1, ))
    assert_size_stride(primals_290, (32, ), (1, ))
    assert_size_stride(primals_291, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_292, (32, ), (1, ))
    assert_size_stride(primals_293, (32, ), (1, ))
    assert_size_stride(primals_294, (32, ), (1, ))
    assert_size_stride(primals_295, (32, ), (1, ))
    assert_size_stride(primals_296, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_297, (32, ), (1, ))
    assert_size_stride(primals_298, (32, ), (1, ))
    assert_size_stride(primals_299, (32, ), (1, ))
    assert_size_stride(primals_300, (32, ), (1, ))
    assert_size_stride(primals_301, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_302, (32, ), (1, ))
    assert_size_stride(primals_303, (32, ), (1, ))
    assert_size_stride(primals_304, (32, ), (1, ))
    assert_size_stride(primals_305, (32, ), (1, ))
    assert_size_stride(primals_306, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_307, (48, ), (1, ))
    assert_size_stride(primals_308, (48, ), (1, ))
    assert_size_stride(primals_309, (48, ), (1, ))
    assert_size_stride(primals_310, (48, ), (1, ))
    assert_size_stride(primals_311, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_312, (64, ), (1, ))
    assert_size_stride(primals_313, (64, ), (1, ))
    assert_size_stride(primals_314, (64, ), (1, ))
    assert_size_stride(primals_315, (64, ), (1, ))
    assert_size_stride(primals_316, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_317, (320, ), (1, ))
    assert_size_stride(primals_318, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_319, (32, ), (1, ))
    assert_size_stride(primals_320, (32, ), (1, ))
    assert_size_stride(primals_321, (32, ), (1, ))
    assert_size_stride(primals_322, (32, ), (1, ))
    assert_size_stride(primals_323, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_324, (32, ), (1, ))
    assert_size_stride(primals_325, (32, ), (1, ))
    assert_size_stride(primals_326, (32, ), (1, ))
    assert_size_stride(primals_327, (32, ), (1, ))
    assert_size_stride(primals_328, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_329, (32, ), (1, ))
    assert_size_stride(primals_330, (32, ), (1, ))
    assert_size_stride(primals_331, (32, ), (1, ))
    assert_size_stride(primals_332, (32, ), (1, ))
    assert_size_stride(primals_333, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_334, (32, ), (1, ))
    assert_size_stride(primals_335, (32, ), (1, ))
    assert_size_stride(primals_336, (32, ), (1, ))
    assert_size_stride(primals_337, (32, ), (1, ))
    assert_size_stride(primals_338, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_339, (48, ), (1, ))
    assert_size_stride(primals_340, (48, ), (1, ))
    assert_size_stride(primals_341, (48, ), (1, ))
    assert_size_stride(primals_342, (48, ), (1, ))
    assert_size_stride(primals_343, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_344, (64, ), (1, ))
    assert_size_stride(primals_345, (64, ), (1, ))
    assert_size_stride(primals_346, (64, ), (1, ))
    assert_size_stride(primals_347, (64, ), (1, ))
    assert_size_stride(primals_348, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_349, (320, ), (1, ))
    assert_size_stride(primals_350, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_351, (32, ), (1, ))
    assert_size_stride(primals_352, (32, ), (1, ))
    assert_size_stride(primals_353, (32, ), (1, ))
    assert_size_stride(primals_354, (32, ), (1, ))
    assert_size_stride(primals_355, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_356, (32, ), (1, ))
    assert_size_stride(primals_357, (32, ), (1, ))
    assert_size_stride(primals_358, (32, ), (1, ))
    assert_size_stride(primals_359, (32, ), (1, ))
    assert_size_stride(primals_360, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_361, (32, ), (1, ))
    assert_size_stride(primals_362, (32, ), (1, ))
    assert_size_stride(primals_363, (32, ), (1, ))
    assert_size_stride(primals_364, (32, ), (1, ))
    assert_size_stride(primals_365, (32, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_366, (32, ), (1, ))
    assert_size_stride(primals_367, (32, ), (1, ))
    assert_size_stride(primals_368, (32, ), (1, ))
    assert_size_stride(primals_369, (32, ), (1, ))
    assert_size_stride(primals_370, (48, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_371, (48, ), (1, ))
    assert_size_stride(primals_372, (48, ), (1, ))
    assert_size_stride(primals_373, (48, ), (1, ))
    assert_size_stride(primals_374, (48, ), (1, ))
    assert_size_stride(primals_375, (64, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_376, (64, ), (1, ))
    assert_size_stride(primals_377, (64, ), (1, ))
    assert_size_stride(primals_378, (64, ), (1, ))
    assert_size_stride(primals_379, (64, ), (1, ))
    assert_size_stride(primals_380, (320, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_381, (320, ), (1, ))
    assert_size_stride(primals_382, (384, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_383, (384, ), (1, ))
    assert_size_stride(primals_384, (384, ), (1, ))
    assert_size_stride(primals_385, (384, ), (1, ))
    assert_size_stride(primals_386, (384, ), (1, ))
    assert_size_stride(primals_387, (256, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_388, (256, ), (1, ))
    assert_size_stride(primals_389, (256, ), (1, ))
    assert_size_stride(primals_390, (256, ), (1, ))
    assert_size_stride(primals_391, (256, ), (1, ))
    assert_size_stride(primals_392, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_393, (256, ), (1, ))
    assert_size_stride(primals_394, (256, ), (1, ))
    assert_size_stride(primals_395, (256, ), (1, ))
    assert_size_stride(primals_396, (256, ), (1, ))
    assert_size_stride(primals_397, (384, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_398, (384, ), (1, ))
    assert_size_stride(primals_399, (384, ), (1, ))
    assert_size_stride(primals_400, (384, ), (1, ))
    assert_size_stride(primals_401, (384, ), (1, ))
    assert_size_stride(primals_402, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_403, (192, ), (1, ))
    assert_size_stride(primals_404, (192, ), (1, ))
    assert_size_stride(primals_405, (192, ), (1, ))
    assert_size_stride(primals_406, (192, ), (1, ))
    assert_size_stride(primals_407, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_408, (128, ), (1, ))
    assert_size_stride(primals_409, (128, ), (1, ))
    assert_size_stride(primals_410, (128, ), (1, ))
    assert_size_stride(primals_411, (128, ), (1, ))
    assert_size_stride(primals_412, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_413, (160, ), (1, ))
    assert_size_stride(primals_414, (160, ), (1, ))
    assert_size_stride(primals_415, (160, ), (1, ))
    assert_size_stride(primals_416, (160, ), (1, ))
    assert_size_stride(primals_417, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_418, (192, ), (1, ))
    assert_size_stride(primals_419, (192, ), (1, ))
    assert_size_stride(primals_420, (192, ), (1, ))
    assert_size_stride(primals_421, (192, ), (1, ))
    assert_size_stride(primals_422, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_423, (1088, ), (1, ))
    assert_size_stride(primals_424, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_425, (192, ), (1, ))
    assert_size_stride(primals_426, (192, ), (1, ))
    assert_size_stride(primals_427, (192, ), (1, ))
    assert_size_stride(primals_428, (192, ), (1, ))
    assert_size_stride(primals_429, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_430, (128, ), (1, ))
    assert_size_stride(primals_431, (128, ), (1, ))
    assert_size_stride(primals_432, (128, ), (1, ))
    assert_size_stride(primals_433, (128, ), (1, ))
    assert_size_stride(primals_434, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_435, (160, ), (1, ))
    assert_size_stride(primals_436, (160, ), (1, ))
    assert_size_stride(primals_437, (160, ), (1, ))
    assert_size_stride(primals_438, (160, ), (1, ))
    assert_size_stride(primals_439, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_440, (192, ), (1, ))
    assert_size_stride(primals_441, (192, ), (1, ))
    assert_size_stride(primals_442, (192, ), (1, ))
    assert_size_stride(primals_443, (192, ), (1, ))
    assert_size_stride(primals_444, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_445, (1088, ), (1, ))
    assert_size_stride(primals_446, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_447, (192, ), (1, ))
    assert_size_stride(primals_448, (192, ), (1, ))
    assert_size_stride(primals_449, (192, ), (1, ))
    assert_size_stride(primals_450, (192, ), (1, ))
    assert_size_stride(primals_451, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_452, (128, ), (1, ))
    assert_size_stride(primals_453, (128, ), (1, ))
    assert_size_stride(primals_454, (128, ), (1, ))
    assert_size_stride(primals_455, (128, ), (1, ))
    assert_size_stride(primals_456, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_457, (160, ), (1, ))
    assert_size_stride(primals_458, (160, ), (1, ))
    assert_size_stride(primals_459, (160, ), (1, ))
    assert_size_stride(primals_460, (160, ), (1, ))
    assert_size_stride(primals_461, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_462, (192, ), (1, ))
    assert_size_stride(primals_463, (192, ), (1, ))
    assert_size_stride(primals_464, (192, ), (1, ))
    assert_size_stride(primals_465, (192, ), (1, ))
    assert_size_stride(primals_466, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_467, (1088, ), (1, ))
    assert_size_stride(primals_468, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_469, (192, ), (1, ))
    assert_size_stride(primals_470, (192, ), (1, ))
    assert_size_stride(primals_471, (192, ), (1, ))
    assert_size_stride(primals_472, (192, ), (1, ))
    assert_size_stride(primals_473, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_474, (128, ), (1, ))
    assert_size_stride(primals_475, (128, ), (1, ))
    assert_size_stride(primals_476, (128, ), (1, ))
    assert_size_stride(primals_477, (128, ), (1, ))
    assert_size_stride(primals_478, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_479, (160, ), (1, ))
    assert_size_stride(primals_480, (160, ), (1, ))
    assert_size_stride(primals_481, (160, ), (1, ))
    assert_size_stride(primals_482, (160, ), (1, ))
    assert_size_stride(primals_483, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_484, (192, ), (1, ))
    assert_size_stride(primals_485, (192, ), (1, ))
    assert_size_stride(primals_486, (192, ), (1, ))
    assert_size_stride(primals_487, (192, ), (1, ))
    assert_size_stride(primals_488, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_489, (1088, ), (1, ))
    assert_size_stride(primals_490, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_491, (192, ), (1, ))
    assert_size_stride(primals_492, (192, ), (1, ))
    assert_size_stride(primals_493, (192, ), (1, ))
    assert_size_stride(primals_494, (192, ), (1, ))
    assert_size_stride(primals_495, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_496, (128, ), (1, ))
    assert_size_stride(primals_497, (128, ), (1, ))
    assert_size_stride(primals_498, (128, ), (1, ))
    assert_size_stride(primals_499, (128, ), (1, ))
    assert_size_stride(primals_500, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_501, (160, ), (1, ))
    assert_size_stride(primals_502, (160, ), (1, ))
    assert_size_stride(primals_503, (160, ), (1, ))
    assert_size_stride(primals_504, (160, ), (1, ))
    assert_size_stride(primals_505, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_506, (192, ), (1, ))
    assert_size_stride(primals_507, (192, ), (1, ))
    assert_size_stride(primals_508, (192, ), (1, ))
    assert_size_stride(primals_509, (192, ), (1, ))
    assert_size_stride(primals_510, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_511, (1088, ), (1, ))
    assert_size_stride(primals_512, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_513, (192, ), (1, ))
    assert_size_stride(primals_514, (192, ), (1, ))
    assert_size_stride(primals_515, (192, ), (1, ))
    assert_size_stride(primals_516, (192, ), (1, ))
    assert_size_stride(primals_517, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_518, (128, ), (1, ))
    assert_size_stride(primals_519, (128, ), (1, ))
    assert_size_stride(primals_520, (128, ), (1, ))
    assert_size_stride(primals_521, (128, ), (1, ))
    assert_size_stride(primals_522, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_523, (160, ), (1, ))
    assert_size_stride(primals_524, (160, ), (1, ))
    assert_size_stride(primals_525, (160, ), (1, ))
    assert_size_stride(primals_526, (160, ), (1, ))
    assert_size_stride(primals_527, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_528, (192, ), (1, ))
    assert_size_stride(primals_529, (192, ), (1, ))
    assert_size_stride(primals_530, (192, ), (1, ))
    assert_size_stride(primals_531, (192, ), (1, ))
    assert_size_stride(primals_532, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_533, (1088, ), (1, ))
    assert_size_stride(primals_534, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_535, (192, ), (1, ))
    assert_size_stride(primals_536, (192, ), (1, ))
    assert_size_stride(primals_537, (192, ), (1, ))
    assert_size_stride(primals_538, (192, ), (1, ))
    assert_size_stride(primals_539, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_540, (128, ), (1, ))
    assert_size_stride(primals_541, (128, ), (1, ))
    assert_size_stride(primals_542, (128, ), (1, ))
    assert_size_stride(primals_543, (128, ), (1, ))
    assert_size_stride(primals_544, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_545, (160, ), (1, ))
    assert_size_stride(primals_546, (160, ), (1, ))
    assert_size_stride(primals_547, (160, ), (1, ))
    assert_size_stride(primals_548, (160, ), (1, ))
    assert_size_stride(primals_549, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_550, (192, ), (1, ))
    assert_size_stride(primals_551, (192, ), (1, ))
    assert_size_stride(primals_552, (192, ), (1, ))
    assert_size_stride(primals_553, (192, ), (1, ))
    assert_size_stride(primals_554, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_555, (1088, ), (1, ))
    assert_size_stride(primals_556, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_557, (192, ), (1, ))
    assert_size_stride(primals_558, (192, ), (1, ))
    assert_size_stride(primals_559, (192, ), (1, ))
    assert_size_stride(primals_560, (192, ), (1, ))
    assert_size_stride(primals_561, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_562, (128, ), (1, ))
    assert_size_stride(primals_563, (128, ), (1, ))
    assert_size_stride(primals_564, (128, ), (1, ))
    assert_size_stride(primals_565, (128, ), (1, ))
    assert_size_stride(primals_566, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_567, (160, ), (1, ))
    assert_size_stride(primals_568, (160, ), (1, ))
    assert_size_stride(primals_569, (160, ), (1, ))
    assert_size_stride(primals_570, (160, ), (1, ))
    assert_size_stride(primals_571, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_572, (192, ), (1, ))
    assert_size_stride(primals_573, (192, ), (1, ))
    assert_size_stride(primals_574, (192, ), (1, ))
    assert_size_stride(primals_575, (192, ), (1, ))
    assert_size_stride(primals_576, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_577, (1088, ), (1, ))
    assert_size_stride(primals_578, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_579, (192, ), (1, ))
    assert_size_stride(primals_580, (192, ), (1, ))
    assert_size_stride(primals_581, (192, ), (1, ))
    assert_size_stride(primals_582, (192, ), (1, ))
    assert_size_stride(primals_583, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_584, (128, ), (1, ))
    assert_size_stride(primals_585, (128, ), (1, ))
    assert_size_stride(primals_586, (128, ), (1, ))
    assert_size_stride(primals_587, (128, ), (1, ))
    assert_size_stride(primals_588, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_589, (160, ), (1, ))
    assert_size_stride(primals_590, (160, ), (1, ))
    assert_size_stride(primals_591, (160, ), (1, ))
    assert_size_stride(primals_592, (160, ), (1, ))
    assert_size_stride(primals_593, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_594, (192, ), (1, ))
    assert_size_stride(primals_595, (192, ), (1, ))
    assert_size_stride(primals_596, (192, ), (1, ))
    assert_size_stride(primals_597, (192, ), (1, ))
    assert_size_stride(primals_598, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_599, (1088, ), (1, ))
    assert_size_stride(primals_600, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_601, (192, ), (1, ))
    assert_size_stride(primals_602, (192, ), (1, ))
    assert_size_stride(primals_603, (192, ), (1, ))
    assert_size_stride(primals_604, (192, ), (1, ))
    assert_size_stride(primals_605, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_606, (128, ), (1, ))
    assert_size_stride(primals_607, (128, ), (1, ))
    assert_size_stride(primals_608, (128, ), (1, ))
    assert_size_stride(primals_609, (128, ), (1, ))
    assert_size_stride(primals_610, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_611, (160, ), (1, ))
    assert_size_stride(primals_612, (160, ), (1, ))
    assert_size_stride(primals_613, (160, ), (1, ))
    assert_size_stride(primals_614, (160, ), (1, ))
    assert_size_stride(primals_615, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_616, (192, ), (1, ))
    assert_size_stride(primals_617, (192, ), (1, ))
    assert_size_stride(primals_618, (192, ), (1, ))
    assert_size_stride(primals_619, (192, ), (1, ))
    assert_size_stride(primals_620, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_621, (1088, ), (1, ))
    assert_size_stride(primals_622, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_623, (192, ), (1, ))
    assert_size_stride(primals_624, (192, ), (1, ))
    assert_size_stride(primals_625, (192, ), (1, ))
    assert_size_stride(primals_626, (192, ), (1, ))
    assert_size_stride(primals_627, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_628, (128, ), (1, ))
    assert_size_stride(primals_629, (128, ), (1, ))
    assert_size_stride(primals_630, (128, ), (1, ))
    assert_size_stride(primals_631, (128, ), (1, ))
    assert_size_stride(primals_632, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_633, (160, ), (1, ))
    assert_size_stride(primals_634, (160, ), (1, ))
    assert_size_stride(primals_635, (160, ), (1, ))
    assert_size_stride(primals_636, (160, ), (1, ))
    assert_size_stride(primals_637, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_638, (192, ), (1, ))
    assert_size_stride(primals_639, (192, ), (1, ))
    assert_size_stride(primals_640, (192, ), (1, ))
    assert_size_stride(primals_641, (192, ), (1, ))
    assert_size_stride(primals_642, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_643, (1088, ), (1, ))
    assert_size_stride(primals_644, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_645, (192, ), (1, ))
    assert_size_stride(primals_646, (192, ), (1, ))
    assert_size_stride(primals_647, (192, ), (1, ))
    assert_size_stride(primals_648, (192, ), (1, ))
    assert_size_stride(primals_649, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_650, (128, ), (1, ))
    assert_size_stride(primals_651, (128, ), (1, ))
    assert_size_stride(primals_652, (128, ), (1, ))
    assert_size_stride(primals_653, (128, ), (1, ))
    assert_size_stride(primals_654, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_655, (160, ), (1, ))
    assert_size_stride(primals_656, (160, ), (1, ))
    assert_size_stride(primals_657, (160, ), (1, ))
    assert_size_stride(primals_658, (160, ), (1, ))
    assert_size_stride(primals_659, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_660, (192, ), (1, ))
    assert_size_stride(primals_661, (192, ), (1, ))
    assert_size_stride(primals_662, (192, ), (1, ))
    assert_size_stride(primals_663, (192, ), (1, ))
    assert_size_stride(primals_664, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_665, (1088, ), (1, ))
    assert_size_stride(primals_666, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_667, (192, ), (1, ))
    assert_size_stride(primals_668, (192, ), (1, ))
    assert_size_stride(primals_669, (192, ), (1, ))
    assert_size_stride(primals_670, (192, ), (1, ))
    assert_size_stride(primals_671, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_672, (128, ), (1, ))
    assert_size_stride(primals_673, (128, ), (1, ))
    assert_size_stride(primals_674, (128, ), (1, ))
    assert_size_stride(primals_675, (128, ), (1, ))
    assert_size_stride(primals_676, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_677, (160, ), (1, ))
    assert_size_stride(primals_678, (160, ), (1, ))
    assert_size_stride(primals_679, (160, ), (1, ))
    assert_size_stride(primals_680, (160, ), (1, ))
    assert_size_stride(primals_681, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_682, (192, ), (1, ))
    assert_size_stride(primals_683, (192, ), (1, ))
    assert_size_stride(primals_684, (192, ), (1, ))
    assert_size_stride(primals_685, (192, ), (1, ))
    assert_size_stride(primals_686, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_687, (1088, ), (1, ))
    assert_size_stride(primals_688, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_689, (192, ), (1, ))
    assert_size_stride(primals_690, (192, ), (1, ))
    assert_size_stride(primals_691, (192, ), (1, ))
    assert_size_stride(primals_692, (192, ), (1, ))
    assert_size_stride(primals_693, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_694, (128, ), (1, ))
    assert_size_stride(primals_695, (128, ), (1, ))
    assert_size_stride(primals_696, (128, ), (1, ))
    assert_size_stride(primals_697, (128, ), (1, ))
    assert_size_stride(primals_698, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_699, (160, ), (1, ))
    assert_size_stride(primals_700, (160, ), (1, ))
    assert_size_stride(primals_701, (160, ), (1, ))
    assert_size_stride(primals_702, (160, ), (1, ))
    assert_size_stride(primals_703, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_704, (192, ), (1, ))
    assert_size_stride(primals_705, (192, ), (1, ))
    assert_size_stride(primals_706, (192, ), (1, ))
    assert_size_stride(primals_707, (192, ), (1, ))
    assert_size_stride(primals_708, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_709, (1088, ), (1, ))
    assert_size_stride(primals_710, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_711, (192, ), (1, ))
    assert_size_stride(primals_712, (192, ), (1, ))
    assert_size_stride(primals_713, (192, ), (1, ))
    assert_size_stride(primals_714, (192, ), (1, ))
    assert_size_stride(primals_715, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_716, (128, ), (1, ))
    assert_size_stride(primals_717, (128, ), (1, ))
    assert_size_stride(primals_718, (128, ), (1, ))
    assert_size_stride(primals_719, (128, ), (1, ))
    assert_size_stride(primals_720, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_721, (160, ), (1, ))
    assert_size_stride(primals_722, (160, ), (1, ))
    assert_size_stride(primals_723, (160, ), (1, ))
    assert_size_stride(primals_724, (160, ), (1, ))
    assert_size_stride(primals_725, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_726, (192, ), (1, ))
    assert_size_stride(primals_727, (192, ), (1, ))
    assert_size_stride(primals_728, (192, ), (1, ))
    assert_size_stride(primals_729, (192, ), (1, ))
    assert_size_stride(primals_730, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_731, (1088, ), (1, ))
    assert_size_stride(primals_732, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_733, (192, ), (1, ))
    assert_size_stride(primals_734, (192, ), (1, ))
    assert_size_stride(primals_735, (192, ), (1, ))
    assert_size_stride(primals_736, (192, ), (1, ))
    assert_size_stride(primals_737, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_738, (128, ), (1, ))
    assert_size_stride(primals_739, (128, ), (1, ))
    assert_size_stride(primals_740, (128, ), (1, ))
    assert_size_stride(primals_741, (128, ), (1, ))
    assert_size_stride(primals_742, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_743, (160, ), (1, ))
    assert_size_stride(primals_744, (160, ), (1, ))
    assert_size_stride(primals_745, (160, ), (1, ))
    assert_size_stride(primals_746, (160, ), (1, ))
    assert_size_stride(primals_747, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_748, (192, ), (1, ))
    assert_size_stride(primals_749, (192, ), (1, ))
    assert_size_stride(primals_750, (192, ), (1, ))
    assert_size_stride(primals_751, (192, ), (1, ))
    assert_size_stride(primals_752, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_753, (1088, ), (1, ))
    assert_size_stride(primals_754, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_755, (192, ), (1, ))
    assert_size_stride(primals_756, (192, ), (1, ))
    assert_size_stride(primals_757, (192, ), (1, ))
    assert_size_stride(primals_758, (192, ), (1, ))
    assert_size_stride(primals_759, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_760, (128, ), (1, ))
    assert_size_stride(primals_761, (128, ), (1, ))
    assert_size_stride(primals_762, (128, ), (1, ))
    assert_size_stride(primals_763, (128, ), (1, ))
    assert_size_stride(primals_764, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_765, (160, ), (1, ))
    assert_size_stride(primals_766, (160, ), (1, ))
    assert_size_stride(primals_767, (160, ), (1, ))
    assert_size_stride(primals_768, (160, ), (1, ))
    assert_size_stride(primals_769, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_770, (192, ), (1, ))
    assert_size_stride(primals_771, (192, ), (1, ))
    assert_size_stride(primals_772, (192, ), (1, ))
    assert_size_stride(primals_773, (192, ), (1, ))
    assert_size_stride(primals_774, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_775, (1088, ), (1, ))
    assert_size_stride(primals_776, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_777, (192, ), (1, ))
    assert_size_stride(primals_778, (192, ), (1, ))
    assert_size_stride(primals_779, (192, ), (1, ))
    assert_size_stride(primals_780, (192, ), (1, ))
    assert_size_stride(primals_781, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_782, (128, ), (1, ))
    assert_size_stride(primals_783, (128, ), (1, ))
    assert_size_stride(primals_784, (128, ), (1, ))
    assert_size_stride(primals_785, (128, ), (1, ))
    assert_size_stride(primals_786, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_787, (160, ), (1, ))
    assert_size_stride(primals_788, (160, ), (1, ))
    assert_size_stride(primals_789, (160, ), (1, ))
    assert_size_stride(primals_790, (160, ), (1, ))
    assert_size_stride(primals_791, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_792, (192, ), (1, ))
    assert_size_stride(primals_793, (192, ), (1, ))
    assert_size_stride(primals_794, (192, ), (1, ))
    assert_size_stride(primals_795, (192, ), (1, ))
    assert_size_stride(primals_796, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_797, (1088, ), (1, ))
    assert_size_stride(primals_798, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_799, (192, ), (1, ))
    assert_size_stride(primals_800, (192, ), (1, ))
    assert_size_stride(primals_801, (192, ), (1, ))
    assert_size_stride(primals_802, (192, ), (1, ))
    assert_size_stride(primals_803, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_804, (128, ), (1, ))
    assert_size_stride(primals_805, (128, ), (1, ))
    assert_size_stride(primals_806, (128, ), (1, ))
    assert_size_stride(primals_807, (128, ), (1, ))
    assert_size_stride(primals_808, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_809, (160, ), (1, ))
    assert_size_stride(primals_810, (160, ), (1, ))
    assert_size_stride(primals_811, (160, ), (1, ))
    assert_size_stride(primals_812, (160, ), (1, ))
    assert_size_stride(primals_813, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_814, (192, ), (1, ))
    assert_size_stride(primals_815, (192, ), (1, ))
    assert_size_stride(primals_816, (192, ), (1, ))
    assert_size_stride(primals_817, (192, ), (1, ))
    assert_size_stride(primals_818, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_819, (1088, ), (1, ))
    assert_size_stride(primals_820, (192, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_821, (192, ), (1, ))
    assert_size_stride(primals_822, (192, ), (1, ))
    assert_size_stride(primals_823, (192, ), (1, ))
    assert_size_stride(primals_824, (192, ), (1, ))
    assert_size_stride(primals_825, (128, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_826, (128, ), (1, ))
    assert_size_stride(primals_827, (128, ), (1, ))
    assert_size_stride(primals_828, (128, ), (1, ))
    assert_size_stride(primals_829, (128, ), (1, ))
    assert_size_stride(primals_830, (160, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_831, (160, ), (1, ))
    assert_size_stride(primals_832, (160, ), (1, ))
    assert_size_stride(primals_833, (160, ), (1, ))
    assert_size_stride(primals_834, (160, ), (1, ))
    assert_size_stride(primals_835, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_836, (192, ), (1, ))
    assert_size_stride(primals_837, (192, ), (1, ))
    assert_size_stride(primals_838, (192, ), (1, ))
    assert_size_stride(primals_839, (192, ), (1, ))
    assert_size_stride(primals_840, (1088, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_841, (1088, ), (1, ))
    assert_size_stride(primals_842, (256, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_843, (256, ), (1, ))
    assert_size_stride(primals_844, (256, ), (1, ))
    assert_size_stride(primals_845, (256, ), (1, ))
    assert_size_stride(primals_846, (256, ), (1, ))
    assert_size_stride(primals_847, (384, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_848, (384, ), (1, ))
    assert_size_stride(primals_849, (384, ), (1, ))
    assert_size_stride(primals_850, (384, ), (1, ))
    assert_size_stride(primals_851, (384, ), (1, ))
    assert_size_stride(primals_852, (256, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_853, (256, ), (1, ))
    assert_size_stride(primals_854, (256, ), (1, ))
    assert_size_stride(primals_855, (256, ), (1, ))
    assert_size_stride(primals_856, (256, ), (1, ))
    assert_size_stride(primals_857, (288, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_858, (288, ), (1, ))
    assert_size_stride(primals_859, (288, ), (1, ))
    assert_size_stride(primals_860, (288, ), (1, ))
    assert_size_stride(primals_861, (288, ), (1, ))
    assert_size_stride(primals_862, (256, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_863, (256, ), (1, ))
    assert_size_stride(primals_864, (256, ), (1, ))
    assert_size_stride(primals_865, (256, ), (1, ))
    assert_size_stride(primals_866, (256, ), (1, ))
    assert_size_stride(primals_867, (288, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_868, (288, ), (1, ))
    assert_size_stride(primals_869, (288, ), (1, ))
    assert_size_stride(primals_870, (288, ), (1, ))
    assert_size_stride(primals_871, (288, ), (1, ))
    assert_size_stride(primals_872, (320, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(primals_873, (320, ), (1, ))
    assert_size_stride(primals_874, (320, ), (1, ))
    assert_size_stride(primals_875, (320, ), (1, ))
    assert_size_stride(primals_876, (320, ), (1, ))
    assert_size_stride(primals_877, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_878, (192, ), (1, ))
    assert_size_stride(primals_879, (192, ), (1, ))
    assert_size_stride(primals_880, (192, ), (1, ))
    assert_size_stride(primals_881, (192, ), (1, ))
    assert_size_stride(primals_882, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_883, (192, ), (1, ))
    assert_size_stride(primals_884, (192, ), (1, ))
    assert_size_stride(primals_885, (192, ), (1, ))
    assert_size_stride(primals_886, (192, ), (1, ))
    assert_size_stride(primals_887, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_888, (224, ), (1, ))
    assert_size_stride(primals_889, (224, ), (1, ))
    assert_size_stride(primals_890, (224, ), (1, ))
    assert_size_stride(primals_891, (224, ), (1, ))
    assert_size_stride(primals_892, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_893, (256, ), (1, ))
    assert_size_stride(primals_894, (256, ), (1, ))
    assert_size_stride(primals_895, (256, ), (1, ))
    assert_size_stride(primals_896, (256, ), (1, ))
    assert_size_stride(primals_897, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_898, (2080, ), (1, ))
    assert_size_stride(primals_899, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_900, (192, ), (1, ))
    assert_size_stride(primals_901, (192, ), (1, ))
    assert_size_stride(primals_902, (192, ), (1, ))
    assert_size_stride(primals_903, (192, ), (1, ))
    assert_size_stride(primals_904, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_905, (192, ), (1, ))
    assert_size_stride(primals_906, (192, ), (1, ))
    assert_size_stride(primals_907, (192, ), (1, ))
    assert_size_stride(primals_908, (192, ), (1, ))
    assert_size_stride(primals_909, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_910, (224, ), (1, ))
    assert_size_stride(primals_911, (224, ), (1, ))
    assert_size_stride(primals_912, (224, ), (1, ))
    assert_size_stride(primals_913, (224, ), (1, ))
    assert_size_stride(primals_914, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_915, (256, ), (1, ))
    assert_size_stride(primals_916, (256, ), (1, ))
    assert_size_stride(primals_917, (256, ), (1, ))
    assert_size_stride(primals_918, (256, ), (1, ))
    assert_size_stride(primals_919, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_920, (2080, ), (1, ))
    assert_size_stride(primals_921, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_922, (192, ), (1, ))
    assert_size_stride(primals_923, (192, ), (1, ))
    assert_size_stride(primals_924, (192, ), (1, ))
    assert_size_stride(primals_925, (192, ), (1, ))
    assert_size_stride(primals_926, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_927, (192, ), (1, ))
    assert_size_stride(primals_928, (192, ), (1, ))
    assert_size_stride(primals_929, (192, ), (1, ))
    assert_size_stride(primals_930, (192, ), (1, ))
    assert_size_stride(primals_931, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_932, (224, ), (1, ))
    assert_size_stride(primals_933, (224, ), (1, ))
    assert_size_stride(primals_934, (224, ), (1, ))
    assert_size_stride(primals_935, (224, ), (1, ))
    assert_size_stride(primals_936, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_937, (256, ), (1, ))
    assert_size_stride(primals_938, (256, ), (1, ))
    assert_size_stride(primals_939, (256, ), (1, ))
    assert_size_stride(primals_940, (256, ), (1, ))
    assert_size_stride(primals_941, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_942, (2080, ), (1, ))
    assert_size_stride(primals_943, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_944, (192, ), (1, ))
    assert_size_stride(primals_945, (192, ), (1, ))
    assert_size_stride(primals_946, (192, ), (1, ))
    assert_size_stride(primals_947, (192, ), (1, ))
    assert_size_stride(primals_948, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_949, (192, ), (1, ))
    assert_size_stride(primals_950, (192, ), (1, ))
    assert_size_stride(primals_951, (192, ), (1, ))
    assert_size_stride(primals_952, (192, ), (1, ))
    assert_size_stride(primals_953, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_954, (224, ), (1, ))
    assert_size_stride(primals_955, (224, ), (1, ))
    assert_size_stride(primals_956, (224, ), (1, ))
    assert_size_stride(primals_957, (224, ), (1, ))
    assert_size_stride(primals_958, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_959, (256, ), (1, ))
    assert_size_stride(primals_960, (256, ), (1, ))
    assert_size_stride(primals_961, (256, ), (1, ))
    assert_size_stride(primals_962, (256, ), (1, ))
    assert_size_stride(primals_963, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_964, (2080, ), (1, ))
    assert_size_stride(primals_965, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_966, (192, ), (1, ))
    assert_size_stride(primals_967, (192, ), (1, ))
    assert_size_stride(primals_968, (192, ), (1, ))
    assert_size_stride(primals_969, (192, ), (1, ))
    assert_size_stride(primals_970, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_971, (192, ), (1, ))
    assert_size_stride(primals_972, (192, ), (1, ))
    assert_size_stride(primals_973, (192, ), (1, ))
    assert_size_stride(primals_974, (192, ), (1, ))
    assert_size_stride(primals_975, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_976, (224, ), (1, ))
    assert_size_stride(primals_977, (224, ), (1, ))
    assert_size_stride(primals_978, (224, ), (1, ))
    assert_size_stride(primals_979, (224, ), (1, ))
    assert_size_stride(primals_980, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_981, (256, ), (1, ))
    assert_size_stride(primals_982, (256, ), (1, ))
    assert_size_stride(primals_983, (256, ), (1, ))
    assert_size_stride(primals_984, (256, ), (1, ))
    assert_size_stride(primals_985, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_986, (2080, ), (1, ))
    assert_size_stride(primals_987, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_988, (192, ), (1, ))
    assert_size_stride(primals_989, (192, ), (1, ))
    assert_size_stride(primals_990, (192, ), (1, ))
    assert_size_stride(primals_991, (192, ), (1, ))
    assert_size_stride(primals_992, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_993, (192, ), (1, ))
    assert_size_stride(primals_994, (192, ), (1, ))
    assert_size_stride(primals_995, (192, ), (1, ))
    assert_size_stride(primals_996, (192, ), (1, ))
    assert_size_stride(primals_997, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_998, (224, ), (1, ))
    assert_size_stride(primals_999, (224, ), (1, ))
    assert_size_stride(primals_1000, (224, ), (1, ))
    assert_size_stride(primals_1001, (224, ), (1, ))
    assert_size_stride(primals_1002, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_1003, (256, ), (1, ))
    assert_size_stride(primals_1004, (256, ), (1, ))
    assert_size_stride(primals_1005, (256, ), (1, ))
    assert_size_stride(primals_1006, (256, ), (1, ))
    assert_size_stride(primals_1007, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_1008, (2080, ), (1, ))
    assert_size_stride(primals_1009, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_1010, (192, ), (1, ))
    assert_size_stride(primals_1011, (192, ), (1, ))
    assert_size_stride(primals_1012, (192, ), (1, ))
    assert_size_stride(primals_1013, (192, ), (1, ))
    assert_size_stride(primals_1014, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_1015, (192, ), (1, ))
    assert_size_stride(primals_1016, (192, ), (1, ))
    assert_size_stride(primals_1017, (192, ), (1, ))
    assert_size_stride(primals_1018, (192, ), (1, ))
    assert_size_stride(primals_1019, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_1020, (224, ), (1, ))
    assert_size_stride(primals_1021, (224, ), (1, ))
    assert_size_stride(primals_1022, (224, ), (1, ))
    assert_size_stride(primals_1023, (224, ), (1, ))
    assert_size_stride(primals_1024, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_1025, (256, ), (1, ))
    assert_size_stride(primals_1026, (256, ), (1, ))
    assert_size_stride(primals_1027, (256, ), (1, ))
    assert_size_stride(primals_1028, (256, ), (1, ))
    assert_size_stride(primals_1029, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_1030, (2080, ), (1, ))
    assert_size_stride(primals_1031, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_1032, (192, ), (1, ))
    assert_size_stride(primals_1033, (192, ), (1, ))
    assert_size_stride(primals_1034, (192, ), (1, ))
    assert_size_stride(primals_1035, (192, ), (1, ))
    assert_size_stride(primals_1036, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_1037, (192, ), (1, ))
    assert_size_stride(primals_1038, (192, ), (1, ))
    assert_size_stride(primals_1039, (192, ), (1, ))
    assert_size_stride(primals_1040, (192, ), (1, ))
    assert_size_stride(primals_1041, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_1042, (224, ), (1, ))
    assert_size_stride(primals_1043, (224, ), (1, ))
    assert_size_stride(primals_1044, (224, ), (1, ))
    assert_size_stride(primals_1045, (224, ), (1, ))
    assert_size_stride(primals_1046, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_1047, (256, ), (1, ))
    assert_size_stride(primals_1048, (256, ), (1, ))
    assert_size_stride(primals_1049, (256, ), (1, ))
    assert_size_stride(primals_1050, (256, ), (1, ))
    assert_size_stride(primals_1051, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_1052, (2080, ), (1, ))
    assert_size_stride(primals_1053, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_1054, (192, ), (1, ))
    assert_size_stride(primals_1055, (192, ), (1, ))
    assert_size_stride(primals_1056, (192, ), (1, ))
    assert_size_stride(primals_1057, (192, ), (1, ))
    assert_size_stride(primals_1058, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_1059, (192, ), (1, ))
    assert_size_stride(primals_1060, (192, ), (1, ))
    assert_size_stride(primals_1061, (192, ), (1, ))
    assert_size_stride(primals_1062, (192, ), (1, ))
    assert_size_stride(primals_1063, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_1064, (224, ), (1, ))
    assert_size_stride(primals_1065, (224, ), (1, ))
    assert_size_stride(primals_1066, (224, ), (1, ))
    assert_size_stride(primals_1067, (224, ), (1, ))
    assert_size_stride(primals_1068, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_1069, (256, ), (1, ))
    assert_size_stride(primals_1070, (256, ), (1, ))
    assert_size_stride(primals_1071, (256, ), (1, ))
    assert_size_stride(primals_1072, (256, ), (1, ))
    assert_size_stride(primals_1073, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_1074, (2080, ), (1, ))
    assert_size_stride(primals_1075, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_1076, (192, ), (1, ))
    assert_size_stride(primals_1077, (192, ), (1, ))
    assert_size_stride(primals_1078, (192, ), (1, ))
    assert_size_stride(primals_1079, (192, ), (1, ))
    assert_size_stride(primals_1080, (192, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_1081, (192, ), (1, ))
    assert_size_stride(primals_1082, (192, ), (1, ))
    assert_size_stride(primals_1083, (192, ), (1, ))
    assert_size_stride(primals_1084, (192, ), (1, ))
    assert_size_stride(primals_1085, (224, 192, 1, 3), (576, 3, 3, 1))
    assert_size_stride(primals_1086, (224, ), (1, ))
    assert_size_stride(primals_1087, (224, ), (1, ))
    assert_size_stride(primals_1088, (224, ), (1, ))
    assert_size_stride(primals_1089, (224, ), (1, ))
    assert_size_stride(primals_1090, (256, 224, 3, 1), (672, 3, 1, 1))
    assert_size_stride(primals_1091, (256, ), (1, ))
    assert_size_stride(primals_1092, (256, ), (1, ))
    assert_size_stride(primals_1093, (256, ), (1, ))
    assert_size_stride(primals_1094, (256, ), (1, ))
    assert_size_stride(primals_1095, (2080, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_1096, (2080, ), (1, ))
    assert_size_stride(primals_1097, (1536, 2080, 1, 1), (2080, 1, 1, 1))
    assert_size_stride(primals_1098, (1536, ), (1, ))
    assert_size_stride(primals_1099, (1536, ), (1, ))
    assert_size_stride(primals_1100, (1536, ), (1, ))
    assert_size_stride(primals_1101, (1536, ), (1, ))
    assert_size_stride(primals_1102, (1001, 1536), (1536, 1))
    assert_size_stride(primals_1103, (1001, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 512, 512), (786432, 1, 1536, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 262144, grid=grid(12, 262144), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_7, buf2, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_7
        buf3 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_12, buf3, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_12
        buf4 = empty_strided_cuda((192, 80, 3, 3), (720, 1, 240, 80), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_22, buf4, 15360, 9, grid=grid(15360, 9), stream=stream0)
        del primals_22
        buf5 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_37, buf5, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_37
        buf6 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_47, buf6, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_47
        buf7 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_52, buf7, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_52
        buf8 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_72, buf8, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_72
        buf9 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_82, buf9, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_82
        buf10 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_87, buf10, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_87
        buf11 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_104, buf11, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_104
        buf12 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_114, buf12, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_114
        buf13 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_119, buf13, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_119
        buf14 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_136, buf14, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_136
        buf15 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_146, buf15, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_146
        buf16 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_151, buf16, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_151
        buf17 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_168, buf17, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_168
        buf18 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_178, buf18, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_178
        buf19 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_183, buf19, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_183
        buf20 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_200, buf20, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_200
        buf21 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_210, buf21, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_210
        buf22 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_215, buf22, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_215
        buf23 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_232, buf23, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_232
        buf24 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_242, buf24, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_242
        buf25 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_247, buf25, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_247
        buf26 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_264, buf26, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_264
        buf27 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_274, buf27, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_274
        buf28 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_279, buf28, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_279
        buf29 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_296, buf29, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_296
        buf30 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_306, buf30, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_306
        buf31 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_311, buf31, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_311
        buf32 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_328, buf32, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_328
        buf33 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_338, buf33, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_338
        buf34 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_343, buf34, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_343
        buf35 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_360, buf35, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_360
        buf36 = empty_strided_cuda((48, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_370, buf36, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del primals_370
        buf37 = empty_strided_cuda((64, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_375, buf37, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_375
        buf38 = empty_strided_cuda((384, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_382, buf38, 122880, 9, grid=grid(122880, 9), stream=stream0)
        del primals_382
        buf39 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_392, buf39, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_392
        buf40 = empty_strided_cuda((384, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_397, buf40, 98304, 9, grid=grid(98304, 9), stream=stream0)
        del primals_397
        buf41 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_412, buf41, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_412
        buf42 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_417, buf42, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_417
        buf43 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_434, buf43, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_434
        buf44 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_439, buf44, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_439
        buf45 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_456, buf45, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_456
        buf46 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_461, buf46, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_461
        buf47 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_478, buf47, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_478
        buf48 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_483, buf48, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_483
        buf49 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_500, buf49, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_500
        buf50 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_505, buf50, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_505
        buf51 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_522, buf51, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_522
        buf52 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_527, buf52, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_527
        buf53 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_544, buf53, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_544
        buf54 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_549, buf54, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_549
        buf55 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_566, buf55, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_566
        buf56 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_571, buf56, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_571
        buf57 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_588, buf57, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_588
        buf58 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_593, buf58, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_593
        buf59 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_610, buf59, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_610
        buf60 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_615, buf60, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_615
        buf61 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_632, buf61, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_632
        buf62 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_637, buf62, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_637
        buf63 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_654, buf63, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_654
        buf64 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_659, buf64, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_659
        buf65 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_676, buf65, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_676
        buf66 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_681, buf66, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_681
        buf67 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_698, buf67, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_698
        buf68 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_703, buf68, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_703
        buf69 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_720, buf69, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_720
        buf70 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_725, buf70, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_725
        buf71 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_742, buf71, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_742
        buf72 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_747, buf72, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_747
        buf73 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_764, buf73, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_764
        buf74 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_769, buf74, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_769
        buf75 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_786, buf75, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_786
        buf76 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_791, buf76, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_791
        buf77 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_808, buf77, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_808
        buf78 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_813, buf78, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_813
        buf79 = empty_strided_cuda((160, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_830, buf79, 20480, 7, grid=grid(20480, 7), stream=stream0)
        del primals_830
        buf80 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_835, buf80, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_835
        buf81 = empty_strided_cuda((384, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_847, buf81, 98304, 9, grid=grid(98304, 9), stream=stream0)
        del primals_847
        buf82 = empty_strided_cuda((288, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_857, buf82, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_857
        buf83 = empty_strided_cuda((288, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_867, buf83, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_867
        buf84 = empty_strided_cuda((320, 288, 3, 3), (2592, 1, 864, 288), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_872, buf84, 92160, 9, grid=grid(92160, 9), stream=stream0)
        del primals_872
        buf85 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_887, buf85, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_887
        buf86 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_892, buf86, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_892
        buf87 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_909, buf87, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_909
        buf88 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_914, buf88, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_914
        buf89 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_931, buf89, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_931
        buf90 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_936, buf90, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_936
        buf91 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_953, buf91, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_953
        buf92 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_958, buf92, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_958
        buf93 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_975, buf93, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_975
        buf94 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_980, buf94, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_980
        buf95 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_997, buf95, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_997
        buf96 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_1002, buf96, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_1002
        buf97 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_1019, buf97, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_1019
        buf98 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_1024, buf98, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_1024
        buf99 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_1041, buf99, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_1041
        buf100 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_1046, buf100, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_1046
        buf101 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_1063, buf101, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_1063
        buf102 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_1068, buf102, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_1068
        buf103 = empty_strided_cuda((224, 192, 1, 3), (576, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_1085, buf103, 43008, 3, grid=grid(43008, 3), stream=stream0)
        del primals_1085
        buf104 = empty_strided_cuda((256, 224, 3, 1), (672, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_1090, buf104, 57344, 3, grid=grid(57344, 3), stream=stream0)
        del primals_1090
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 32, 255, 255), (2080800, 1, 8160, 32))
        buf106 = empty_strided_cuda((4, 32, 255, 255), (2080800, 1, 8160, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf105, primals_3, primals_4, primals_5, primals_6, buf106, 8323200, grid=grid(8323200), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, buf2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 32, 253, 253), (2048288, 1, 8096, 32))
        buf108 = empty_strided_cuda((4, 32, 253, 253), (2048288, 1, 8096, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf107, primals_8, primals_9, primals_10, primals_11, buf108, 8193152, grid=grid(8193152), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 64, 253, 253), (4096576, 1, 16192, 64))
        buf110 = empty_strided_cuda((4, 64, 253, 253), (4096576, 1, 16192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf109, primals_13, primals_14, primals_15, primals_16, buf110, 16386304, grid=grid(16386304), stream=stream0)
        del primals_16
        buf111 = empty_strided_cuda((4, 64, 126, 126), (1016064, 1, 8064, 64), torch.float32)
        buf112 = empty_strided_cuda((4, 64, 126, 126), (1016064, 1, 8064, 64), torch.int8)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_22.run(buf110, buf111, buf112, 4064256, grid=grid(4064256), stream=stream0)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf111, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 80, 126, 126), (1270080, 1, 10080, 80))
        buf114 = empty_strided_cuda((4, 80, 126, 126), (1270080, 1, 10080, 80), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf113, primals_18, primals_19, primals_20, primals_21, buf114, 5080320, grid=grid(5080320), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, buf4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 192, 124, 124), (2952192, 1, 23808, 192))
        buf116 = empty_strided_cuda((4, 192, 124, 124), (2952192, 1, 23808, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_14, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf115, primals_23, primals_24, primals_25, primals_26, buf116, 11808768, grid=grid(11808768), stream=stream0)
        del primals_26
        buf117 = empty_strided_cuda((4, 192, 61, 61), (714432, 1, 11712, 192), torch.float32)
        buf118 = empty_strided_cuda((4, 192, 61, 61), (714432, 1, 11712, 192), torch.int8)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_25.run(buf116, buf117, buf118, 2857728, grid=grid(2857728), stream=stream0)
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf117, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 96, 61, 61), (357216, 1, 5856, 96))
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf117, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf121 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_21, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf120, primals_33, primals_34, primals_35, primals_36, buf121, 714432, grid=grid(714432), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, buf5, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 64, 61, 61), (238144, 1, 3904, 64))
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf117, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf124 = empty_strided_cuda((4, 64, 61, 61), (238144, 1, 3904, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf123, primals_43, primals_44, primals_45, primals_46, buf124, 952576, grid=grid(952576), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 96, 61, 61), (357216, 1, 5856, 96))
        buf126 = empty_strided_cuda((4, 96, 61, 61), (357216, 1, 5856, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf125, primals_48, primals_49, primals_50, primals_51, buf126, 1428864, grid=grid(1428864), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 96, 61, 61), (357216, 1, 5856, 96))
        buf128 = empty_strided_cuda((4, 192, 61, 61), (714432, 1, 11712, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_29.run(buf117, buf128, 2857728, grid=grid(2857728), stream=stream0)
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf130 = empty_strided_cuda((4, 320, 61, 61), (1190720, 1, 19520, 320), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_30.run(buf119, primals_28, primals_29, primals_30, primals_31, buf122, primals_38, primals_39, primals_40, primals_41, buf127, primals_53, primals_54, primals_55, primals_56, buf129, primals_58, primals_59, primals_60, primals_61, buf130, 4762880, grid=grid(4762880), stream=stream0)
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf130, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf133 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_42, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf132, primals_68, primals_69, primals_70, primals_71, buf133, 476288, grid=grid(476288), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf130, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf136 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_48, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf135, primals_78, primals_79, primals_80, primals_81, buf136, 476288, grid=grid(476288), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf138 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_51, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf137, primals_83, primals_84, primals_85, primals_86, buf138, 714432, grid=grid(714432), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf140 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf131, primals_63, primals_64, primals_65, primals_66, buf134, primals_73, primals_74, primals_75, primals_76, buf139, primals_88, primals_89, primals_90, primals_91, buf140, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf142 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [out_2, mul, out_3, out_4], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf142, primals_93, buf130, 4762880, grid=grid(4762880), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf142, primals_99, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf145 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_60, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf144, primals_100, primals_101, primals_102, primals_103, buf145, 476288, grid=grid(476288), stream=stream0)
        del primals_103
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf142, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf148 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_66, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf147, primals_110, primals_111, primals_112, primals_113, buf148, 476288, grid=grid(476288), stream=stream0)
        del primals_113
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf150 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_69, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf149, primals_115, primals_116, primals_117, primals_118, buf150, 714432, grid=grid(714432), stream=stream0)
        del primals_118
        # Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf152 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf143, primals_95, primals_96, primals_97, primals_98, buf146, primals_105, primals_106, primals_107, primals_108, buf151, primals_120, primals_121, primals_122, primals_123, buf152, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [out_6, mul_1, out_7, out_8], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf154, primals_125, buf142, 4762880, grid=grid(4762880), stream=stream0)
        del primals_125
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf154, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf157 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_78, x_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf156, primals_132, primals_133, primals_134, primals_135, buf157, 476288, grid=grid(476288), stream=stream0)
        del primals_135
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf154, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf160 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_84, x_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf159, primals_142, primals_143, primals_144, primals_145, buf160, 476288, grid=grid(476288), stream=stream0)
        del primals_145
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf162 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_87, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf161, primals_147, primals_148, primals_149, primals_150, buf162, 714432, grid=grid(714432), stream=stream0)
        del primals_150
        # Topologically Sorted Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf164 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf155, primals_127, primals_128, primals_129, primals_130, buf158, primals_137, primals_138, primals_139, primals_140, buf163, primals_152, primals_153, primals_154, primals_155, buf164, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf166 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [out_10, mul_2, out_11, out_12], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf166, primals_157, buf154, 4762880, grid=grid(4762880), stream=stream0)
        del primals_157
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf166, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf169 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_96, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf168, primals_164, primals_165, primals_166, primals_167, buf169, 476288, grid=grid(476288), stream=stream0)
        del primals_167
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf166, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf172 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_102, x_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf171, primals_174, primals_175, primals_176, primals_177, buf172, 476288, grid=grid(476288), stream=stream0)
        del primals_177
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf174 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_105, x_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf173, primals_179, primals_180, primals_181, primals_182, buf174, 714432, grid=grid(714432), stream=stream0)
        del primals_182
        # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf176 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf167, primals_159, primals_160, primals_161, primals_162, buf170, primals_169, primals_170, primals_171, primals_172, buf175, primals_184, primals_185, primals_186, primals_187, buf176, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf178 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [out_14, mul_3, out_15, out_16], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf178, primals_189, buf166, 4762880, grid=grid(4762880), stream=stream0)
        del primals_189
        # Topologically Sorted Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_113], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf178, primals_195, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf181 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_114, x_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf180, primals_196, primals_197, primals_198, primals_199, buf181, 476288, grid=grid(476288), stream=stream0)
        del primals_199
        # Topologically Sorted Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_119], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf178, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf184 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_120, x_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf183, primals_206, primals_207, primals_208, primals_209, buf184, 476288, grid=grid(476288), stream=stream0)
        del primals_209
        # Topologically Sorted Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf186 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_123, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf185, primals_211, primals_212, primals_213, primals_214, buf186, 714432, grid=grid(714432), stream=stream0)
        del primals_214
        # Topologically Sorted Source Nodes: [x_125], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf188 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf179, primals_191, primals_192, primals_193, primals_194, buf182, primals_201, primals_202, primals_203, primals_204, buf187, primals_216, primals_217, primals_218, primals_219, buf188, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_18], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf190 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [out_18, mul_4, out_19, out_20], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf190, primals_221, buf178, 4762880, grid=grid(4762880), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_131], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf190, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf193 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_132, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf192, primals_228, primals_229, primals_230, primals_231, buf193, 476288, grid=grid(476288), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf190, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf196 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_138, x_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf195, primals_238, primals_239, primals_240, primals_241, buf196, 476288, grid=grid(476288), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf198 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_141, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf197, primals_243, primals_244, primals_245, primals_246, buf198, 714432, grid=grid(714432), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf200 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf191, primals_223, primals_224, primals_225, primals_226, buf194, primals_233, primals_234, primals_235, primals_236, buf199, primals_248, primals_249, primals_250, primals_251, buf200, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf202 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [out_22, mul_5, out_23, out_24], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf202, primals_253, buf190, 4762880, grid=grid(4762880), stream=stream0)
        del primals_253
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf202, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf205 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_150, x_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf204, primals_260, primals_261, primals_262, primals_263, buf205, 476288, grid=grid(476288), stream=stream0)
        del primals_263
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf202, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf208 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf207, primals_270, primals_271, primals_272, primals_273, buf208, 476288, grid=grid(476288), stream=stream0)
        del primals_273
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf210 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf209, primals_275, primals_276, primals_277, primals_278, buf210, 714432, grid=grid(714432), stream=stream0)
        del primals_278
        # Topologically Sorted Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf212 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_25], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf203, primals_255, primals_256, primals_257, primals_258, buf206, primals_265, primals_266, primals_267, primals_268, buf211, primals_280, primals_281, primals_282, primals_283, buf212, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_284, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf214 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [out_26, mul_6, out_27, out_28], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf214, primals_285, buf202, 4762880, grid=grid(4762880), stream=stream0)
        del primals_285
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_167], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf214, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf217 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_168, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf216, primals_292, primals_293, primals_294, primals_295, buf217, 476288, grid=grid(476288), stream=stream0)
        del primals_295
        # Topologically Sorted Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_173], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf214, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf220 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_174, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf219, primals_302, primals_303, primals_304, primals_305, buf220, 476288, grid=grid(476288), stream=stream0)
        del primals_305
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf222 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_177, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf221, primals_307, primals_308, primals_309, primals_310, buf222, 714432, grid=grid(714432), stream=stream0)
        del primals_310
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf224 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_29], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf215, primals_287, primals_288, primals_289, primals_290, buf218, primals_297, primals_298, primals_299, primals_300, buf223, primals_312, primals_313, primals_314, primals_315, buf224, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf226 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [out_30, mul_7, out_31, out_32], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf226, primals_317, buf214, 4762880, grid=grid(4762880), stream=stream0)
        del primals_317
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_318, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_185], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf226, primals_323, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf229 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_186, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf228, primals_324, primals_325, primals_326, primals_327, buf229, 476288, grid=grid(476288), stream=stream0)
        del primals_327
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf226, primals_333, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf232 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf231, primals_334, primals_335, primals_336, primals_337, buf232, 476288, grid=grid(476288), stream=stream0)
        del primals_337
        # Topologically Sorted Source Nodes: [x_194], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf234 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_195, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf233, primals_339, primals_340, primals_341, primals_342, buf234, 714432, grid=grid(714432), stream=stream0)
        del primals_342
        # Topologically Sorted Source Nodes: [x_197], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf236 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf227, primals_319, primals_320, primals_321, primals_322, buf230, primals_329, primals_330, primals_331, primals_332, buf235, primals_344, primals_345, primals_346, primals_347, buf236, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_348, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf238 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [out_34, mul_8, out_35, out_36], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf238, primals_349, buf226, 4762880, grid=grid(4762880), stream=stream0)
        del primals_349
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_350, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_203], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf238, primals_355, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf241 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf240, primals_356, primals_357, primals_358, primals_359, buf241, 476288, grid=grid(476288), stream=stream0)
        del primals_359
        # Topologically Sorted Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 32, 61, 61), (119072, 1, 1952, 32))
        # Topologically Sorted Source Nodes: [x_209], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf238, primals_365, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf244 = empty_strided_cuda((4, 32, 61, 61), (119072, 1, 1952, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_210, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf243, primals_366, primals_367, primals_368, primals_369, buf244, 476288, grid=grid(476288), stream=stream0)
        del primals_369
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, buf36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf246 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf245, primals_371, primals_372, primals_373, primals_374, buf246, 714432, grid=grid(714432), stream=stream0)
        del primals_374
        # Topologically Sorted Source Nodes: [x_215], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf248 = empty_strided_cuda((4, 128, 61, 61), (476288, 1, 7808, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf239, primals_351, primals_352, primals_353, primals_354, buf242, primals_361, primals_362, primals_363, primals_364, buf247, primals_376, primals_377, primals_378, primals_379, buf248, 1905152, grid=grid(1905152), stream=stream0)
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, primals_380, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (4, 320, 61, 61), (1190720, 1, 19520, 320))
        buf250 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [out_38, mul_9, out_39, out_40], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_33.run(buf250, primals_381, buf238, 4762880, grid=grid(4762880), stream=stream0)
        del primals_381
        # Topologically Sorted Source Nodes: [x_218], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, buf38, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 384, 30, 30), (345600, 1, 11520, 384))
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf250, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 256, 61, 61), (952576, 1, 15616, 256))
        buf253 = empty_strided_cuda((4, 256, 61, 61), (952576, 1, 15616, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_222, x_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf252, primals_388, primals_389, primals_390, primals_391, buf253, 3810304, grid=grid(3810304), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 256, 61, 61), (952576, 1, 15616, 256))
        buf255 = empty_strided_cuda((4, 256, 61, 61), (952576, 1, 15616, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_225, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf254, primals_393, primals_394, primals_395, primals_396, buf255, 3810304, grid=grid(3810304), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, buf40, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 384, 30, 30), (345600, 1, 11520, 384))
        buf261 = empty_strided_cuda((4, 1088, 30, 30), (979200, 900, 30, 1), torch.float32)
        buf257 = reinterpret_tensor(buf261, (4, 320, 30, 30), (979200, 900, 30, 1), 691200)  # alias
        buf258 = empty_strided_cuda((4, 320, 30, 30), (288000, 1, 9600, 320), torch.int8)
        # Topologically Sorted Source Nodes: [x2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_35.run(buf250, buf257, buf258, 3600, 320, grid=grid(3600, 320), stream=stream0)
        buf259 = reinterpret_tensor(buf261, (4, 384, 30, 30), (979200, 900, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf251, primals_383, primals_384, primals_385, primals_386, buf259, 1536, 900, grid=grid(1536, 900), stream=stream0)
        buf260 = reinterpret_tensor(buf261, (4, 384, 30, 30), (979200, 900, 30, 1), 345600)  # alias
        # Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf256, primals_398, primals_399, primals_400, primals_401, buf260, 1536, 900, grid=grid(1536, 900), stream=stream0)
        buf262 = empty_strided_cuda((4, 1088, 30, 30), (979200, 1, 32640, 1088), torch.float32)
        # Topologically Sorted Source Nodes: [out_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_37.run(buf261, buf262, 4352, 900, grid=grid(4352, 900), stream=stream0)
        del buf257
        del buf259
        del buf260
        del buf261
        # Topologically Sorted Source Nodes: [x_230], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_233], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf262, primals_407, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf265 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf264, primals_408, primals_409, primals_410, primals_411, buf265, 460800, grid=grid(460800), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, buf41, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf267 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf266, primals_413, primals_414, primals_415, primals_416, buf267, 576000, grid=grid(576000), stream=stream0)
        del primals_416
        # Topologically Sorted Source Nodes: [x_239], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, buf42, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf269 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf263, primals_403, primals_404, primals_405, primals_406, buf268, primals_418, primals_419, primals_420, primals_421, buf269, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf271 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [out_43, mul_10, out_44, out_45], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf271, primals_423, buf262, 3916800, grid=grid(3916800), stream=stream0)
        del primals_423
        # Topologically Sorted Source Nodes: [x_242], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_424, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf271, primals_429, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf274 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_246, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf273, primals_430, primals_431, primals_432, primals_433, buf274, 460800, grid=grid(460800), stream=stream0)
        del primals_433
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, buf43, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf276 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_249, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf275, primals_435, primals_436, primals_437, primals_438, buf276, 576000, grid=grid(576000), stream=stream0)
        del primals_438
        # Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, buf44, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf278 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf272, primals_425, primals_426, primals_427, primals_428, buf277, primals_440, primals_441, primals_442, primals_443, buf278, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_47], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, primals_444, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf280 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [out_47, mul_11, out_48, out_49], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf280, primals_445, buf271, 3916800, grid=grid(3916800), stream=stream0)
        del primals_445
        # Topologically Sorted Source Nodes: [x_254], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_446, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf280, primals_451, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf283 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_258, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf282, primals_452, primals_453, primals_454, primals_455, buf283, 460800, grid=grid(460800), stream=stream0)
        del primals_455
        # Topologically Sorted Source Nodes: [x_260], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, buf45, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf285 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf284, primals_457, primals_458, primals_459, primals_460, buf285, 576000, grid=grid(576000), stream=stream0)
        del primals_460
        # Topologically Sorted Source Nodes: [x_263], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, buf46, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf287 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf281, primals_447, primals_448, primals_449, primals_450, buf286, primals_462, primals_463, primals_464, primals_465, buf287, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_51], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_466, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf289 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [out_51, mul_12, out_52, out_53], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf289, primals_467, buf280, 3916800, grid=grid(3916800), stream=stream0)
        del primals_467
        # Topologically Sorted Source Nodes: [x_266], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, primals_468, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_269], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf289, primals_473, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf292 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf291, primals_474, primals_475, primals_476, primals_477, buf292, 460800, grid=grid(460800), stream=stream0)
        del primals_477
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, buf47, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf294 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_273, x_274], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf293, primals_479, primals_480, primals_481, primals_482, buf294, 576000, grid=grid(576000), stream=stream0)
        del primals_482
        # Topologically Sorted Source Nodes: [x_275], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, buf48, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf296 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf290, primals_469, primals_470, primals_471, primals_472, buf295, primals_484, primals_485, primals_486, primals_487, buf296, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_55], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, primals_488, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf298 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [out_55, mul_13, out_56, out_57], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf298, primals_489, buf289, 3916800, grid=grid(3916800), stream=stream0)
        del primals_489
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_490, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf298, primals_495, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf301 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_282, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf300, primals_496, primals_497, primals_498, primals_499, buf301, 460800, grid=grid(460800), stream=stream0)
        del primals_499
        # Topologically Sorted Source Nodes: [x_284], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, buf49, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf303 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf302, primals_501, primals_502, primals_503, primals_504, buf303, 576000, grid=grid(576000), stream=stream0)
        del primals_504
        # Topologically Sorted Source Nodes: [x_287], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, buf50, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf305 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_58], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf299, primals_491, primals_492, primals_493, primals_494, buf304, primals_506, primals_507, primals_508, primals_509, buf305, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_59], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_510, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf307 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [out_59, mul_14, out_60, out_61], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf307, primals_511, buf298, 3916800, grid=grid(3916800), stream=stream0)
        del primals_511
        # Topologically Sorted Source Nodes: [x_290], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_293], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf307, primals_517, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf310 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf309, primals_518, primals_519, primals_520, primals_521, buf310, 460800, grid=grid(460800), stream=stream0)
        del primals_521
        # Topologically Sorted Source Nodes: [x_296], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, buf51, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf312 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf311, primals_523, primals_524, primals_525, primals_526, buf312, 576000, grid=grid(576000), stream=stream0)
        del primals_526
        # Topologically Sorted Source Nodes: [x_299], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, buf52, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf314 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_62], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf308, primals_513, primals_514, primals_515, primals_516, buf313, primals_528, primals_529, primals_530, primals_531, buf314, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_63], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, primals_532, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf316 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [out_63, mul_15, out_64, out_65], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf316, primals_533, buf307, 3916800, grid=grid(3916800), stream=stream0)
        del primals_533
        # Topologically Sorted Source Nodes: [x_302], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_534, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_305], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf316, primals_539, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf319 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_306, x_307], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf318, primals_540, primals_541, primals_542, primals_543, buf319, 460800, grid=grid(460800), stream=stream0)
        del primals_543
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, buf53, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf321 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_309, x_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf320, primals_545, primals_546, primals_547, primals_548, buf321, 576000, grid=grid(576000), stream=stream0)
        del primals_548
        # Topologically Sorted Source Nodes: [x_311], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, buf54, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf323 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_66], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf317, primals_535, primals_536, primals_537, primals_538, buf322, primals_550, primals_551, primals_552, primals_553, buf323, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_67], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_554, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf325 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [out_67, mul_16, out_68, out_69], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf325, primals_555, buf316, 3916800, grid=grid(3916800), stream=stream0)
        del primals_555
        # Topologically Sorted Source Nodes: [x_314], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_556, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_317], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf325, primals_561, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf328 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_318, x_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf327, primals_562, primals_563, primals_564, primals_565, buf328, 460800, grid=grid(460800), stream=stream0)
        del primals_565
        # Topologically Sorted Source Nodes: [x_320], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, buf55, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf330 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_321, x_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf329, primals_567, primals_568, primals_569, primals_570, buf330, 576000, grid=grid(576000), stream=stream0)
        del primals_570
        # Topologically Sorted Source Nodes: [x_323], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, buf56, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf332 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf326, primals_557, primals_558, primals_559, primals_560, buf331, primals_572, primals_573, primals_574, primals_575, buf332, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_71], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_576, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf334 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [out_71, mul_17, out_72, out_73], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf334, primals_577, buf325, 3916800, grid=grid(3916800), stream=stream0)
        del primals_577
        # Topologically Sorted Source Nodes: [x_326], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf334, primals_578, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_329], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf334, primals_583, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf337 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_330, x_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf336, primals_584, primals_585, primals_586, primals_587, buf337, 460800, grid=grid(460800), stream=stream0)
        del primals_587
        # Topologically Sorted Source Nodes: [x_332], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, buf57, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf339 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_333, x_334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf338, primals_589, primals_590, primals_591, primals_592, buf339, 576000, grid=grid(576000), stream=stream0)
        del primals_592
        # Topologically Sorted Source Nodes: [x_335], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, buf58, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf341 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_74], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf335, primals_579, primals_580, primals_581, primals_582, buf340, primals_594, primals_595, primals_596, primals_597, buf341, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_75], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_598, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf343 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [out_75, mul_18, out_76, out_77], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf343, primals_599, buf334, 3916800, grid=grid(3916800), stream=stream0)
        del primals_599
        # Topologically Sorted Source Nodes: [x_338], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_600, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_341], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf343, primals_605, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf346 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_342, x_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf345, primals_606, primals_607, primals_608, primals_609, buf346, 460800, grid=grid(460800), stream=stream0)
        del primals_609
        # Topologically Sorted Source Nodes: [x_344], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, buf59, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf348 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_345, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf347, primals_611, primals_612, primals_613, primals_614, buf348, 576000, grid=grid(576000), stream=stream0)
        del primals_614
        # Topologically Sorted Source Nodes: [x_347], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, buf60, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf350 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_78], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf344, primals_601, primals_602, primals_603, primals_604, buf349, primals_616, primals_617, primals_618, primals_619, buf350, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_79], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, primals_620, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf352 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [out_79, mul_19, out_80, out_81], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf352, primals_621, buf343, 3916800, grid=grid(3916800), stream=stream0)
        del primals_621
        # Topologically Sorted Source Nodes: [x_350], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_622, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf352, primals_627, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf355 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_354, x_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf354, primals_628, primals_629, primals_630, primals_631, buf355, 460800, grid=grid(460800), stream=stream0)
        del primals_631
        # Topologically Sorted Source Nodes: [x_356], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, buf61, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf357 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_357, x_358], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf356, primals_633, primals_634, primals_635, primals_636, buf357, 576000, grid=grid(576000), stream=stream0)
        del primals_636
        # Topologically Sorted Source Nodes: [x_359], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, buf62, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf359 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_82], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf353, primals_623, primals_624, primals_625, primals_626, buf358, primals_638, primals_639, primals_640, primals_641, buf359, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, primals_642, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf361 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [out_83, mul_20, out_84, out_85], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf361, primals_643, buf352, 3916800, grid=grid(3916800), stream=stream0)
        del primals_643
        # Topologically Sorted Source Nodes: [x_362], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_644, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_365], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf361, primals_649, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf364 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_366, x_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf363, primals_650, primals_651, primals_652, primals_653, buf364, 460800, grid=grid(460800), stream=stream0)
        del primals_653
        # Topologically Sorted Source Nodes: [x_368], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, buf63, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf366 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_369, x_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf365, primals_655, primals_656, primals_657, primals_658, buf366, 576000, grid=grid(576000), stream=stream0)
        del primals_658
        # Topologically Sorted Source Nodes: [x_371], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, buf64, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf368 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_86], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf362, primals_645, primals_646, primals_647, primals_648, buf367, primals_660, primals_661, primals_662, primals_663, buf368, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_87], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, primals_664, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf370 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [out_87, mul_21, out_88, out_89], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf370, primals_665, buf361, 3916800, grid=grid(3916800), stream=stream0)
        del primals_665
        # Topologically Sorted Source Nodes: [x_374], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_666, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_377], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf370, primals_671, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf373 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_378, x_379], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf372, primals_672, primals_673, primals_674, primals_675, buf373, 460800, grid=grid(460800), stream=stream0)
        del primals_675
        # Topologically Sorted Source Nodes: [x_380], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, buf65, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf375 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_381, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf374, primals_677, primals_678, primals_679, primals_680, buf375, 576000, grid=grid(576000), stream=stream0)
        del primals_680
        # Topologically Sorted Source Nodes: [x_383], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, buf66, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf377 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_90], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf371, primals_667, primals_668, primals_669, primals_670, buf376, primals_682, primals_683, primals_684, primals_685, buf377, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_91], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_686, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf379 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [out_91, mul_22, out_92, out_93], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf379, primals_687, buf370, 3916800, grid=grid(3916800), stream=stream0)
        del primals_687
        # Topologically Sorted Source Nodes: [x_386], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, primals_688, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_389], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf379, primals_693, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf382 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_390, x_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf381, primals_694, primals_695, primals_696, primals_697, buf382, 460800, grid=grid(460800), stream=stream0)
        del primals_697
        # Topologically Sorted Source Nodes: [x_392], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, buf67, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf384 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_393, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf383, primals_699, primals_700, primals_701, primals_702, buf384, 576000, grid=grid(576000), stream=stream0)
        del primals_702
        # Topologically Sorted Source Nodes: [x_395], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, buf68, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf386 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_94], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf380, primals_689, primals_690, primals_691, primals_692, buf385, primals_704, primals_705, primals_706, primals_707, buf386, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_95], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf386, primals_708, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf388 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [out_95, mul_23, out_96, out_97], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf388, primals_709, buf379, 3916800, grid=grid(3916800), stream=stream0)
        del primals_709
        # Topologically Sorted Source Nodes: [x_398], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_710, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_401], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf388, primals_715, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf391 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf390, primals_716, primals_717, primals_718, primals_719, buf391, 460800, grid=grid(460800), stream=stream0)
        del primals_719
        # Topologically Sorted Source Nodes: [x_404], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, buf69, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf393 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_405, x_406], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf392, primals_721, primals_722, primals_723, primals_724, buf393, 576000, grid=grid(576000), stream=stream0)
        del primals_724
        # Topologically Sorted Source Nodes: [x_407], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, buf70, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf395 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_98], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf389, primals_711, primals_712, primals_713, primals_714, buf394, primals_726, primals_727, primals_728, primals_729, buf395, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_99], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, primals_730, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf397 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [out_99, mul_24, out_100, out_101], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf397, primals_731, buf388, 3916800, grid=grid(3916800), stream=stream0)
        del primals_731
        # Topologically Sorted Source Nodes: [x_410], Original ATen: [aten.convolution]
        buf398 = extern_kernels.convolution(buf397, primals_732, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_413], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf397, primals_737, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf400 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_414, x_415], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf399, primals_738, primals_739, primals_740, primals_741, buf400, 460800, grid=grid(460800), stream=stream0)
        del primals_741
        # Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, buf71, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf402 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_417, x_418], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf401, primals_743, primals_744, primals_745, primals_746, buf402, 576000, grid=grid(576000), stream=stream0)
        del primals_746
        # Topologically Sorted Source Nodes: [x_419], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, buf72, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf404 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_102], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf398, primals_733, primals_734, primals_735, primals_736, buf403, primals_748, primals_749, primals_750, primals_751, buf404, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(buf404, primals_752, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf406 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [out_103, mul_25, out_104, out_105], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf406, primals_753, buf397, 3916800, grid=grid(3916800), stream=stream0)
        del primals_753
        # Topologically Sorted Source Nodes: [x_422], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf406, primals_754, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_425], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf406, primals_759, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf409 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_426, x_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf408, primals_760, primals_761, primals_762, primals_763, buf409, 460800, grid=grid(460800), stream=stream0)
        del primals_763
        # Topologically Sorted Source Nodes: [x_428], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, buf73, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf411 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_429, x_430], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf410, primals_765, primals_766, primals_767, primals_768, buf411, 576000, grid=grid(576000), stream=stream0)
        del primals_768
        # Topologically Sorted Source Nodes: [x_431], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, buf74, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf413 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf407, primals_755, primals_756, primals_757, primals_758, buf412, primals_770, primals_771, primals_772, primals_773, buf413, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_107], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_774, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf415 = buf414; del buf414  # reuse
        # Topologically Sorted Source Nodes: [out_107, mul_26, out_108, out_109], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf415, primals_775, buf406, 3916800, grid=grid(3916800), stream=stream0)
        del primals_775
        # Topologically Sorted Source Nodes: [x_434], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_776, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_437], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf415, primals_781, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf418 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_438, x_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf417, primals_782, primals_783, primals_784, primals_785, buf418, 460800, grid=grid(460800), stream=stream0)
        del primals_785
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, buf75, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf420 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_441, x_442], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf419, primals_787, primals_788, primals_789, primals_790, buf420, 576000, grid=grid(576000), stream=stream0)
        del primals_790
        # Topologically Sorted Source Nodes: [x_443], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf420, buf76, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf422 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf416, primals_777, primals_778, primals_779, primals_780, buf421, primals_792, primals_793, primals_794, primals_795, buf422, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_111], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf422, primals_796, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf423, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf424 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [out_111, mul_27, out_112, out_113], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf424, primals_797, buf415, 3916800, grid=grid(3916800), stream=stream0)
        del primals_797
        # Topologically Sorted Source Nodes: [x_446], Original ATen: [aten.convolution]
        buf425 = extern_kernels.convolution(buf424, primals_798, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf425, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_449], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf424, primals_803, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf427 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_450, x_451], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf426, primals_804, primals_805, primals_806, primals_807, buf427, 460800, grid=grid(460800), stream=stream0)
        del primals_807
        # Topologically Sorted Source Nodes: [x_452], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, buf77, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf429 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_453, x_454], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf428, primals_809, primals_810, primals_811, primals_812, buf429, 576000, grid=grid(576000), stream=stream0)
        del primals_812
        # Topologically Sorted Source Nodes: [x_455], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, buf78, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf431 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_114], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf425, primals_799, primals_800, primals_801, primals_802, buf430, primals_814, primals_815, primals_816, primals_817, buf431, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_115], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_818, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf433 = buf432; del buf432  # reuse
        # Topologically Sorted Source Nodes: [out_115, mul_28, out_116, out_117], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf433, primals_819, buf424, 3916800, grid=grid(3916800), stream=stream0)
        del primals_819
        # Topologically Sorted Source Nodes: [x_458], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, primals_820, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [x_461], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf433, primals_825, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf436 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_462, x_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf435, primals_826, primals_827, primals_828, primals_829, buf436, 460800, grid=grid(460800), stream=stream0)
        del primals_829
        # Topologically Sorted Source Nodes: [x_464], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, buf79, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf438 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_465, x_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf437, primals_831, primals_832, primals_833, primals_834, buf438, 576000, grid=grid(576000), stream=stream0)
        del primals_834
        # Topologically Sorted Source Nodes: [x_467], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, buf80, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf440 = empty_strided_cuda((4, 384, 30, 30), (345600, 1, 11520, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_118], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf434, primals_821, primals_822, primals_823, primals_824, buf439, primals_836, primals_837, primals_838, primals_839, buf440, 1382400, grid=grid(1382400), stream=stream0)
        # Topologically Sorted Source Nodes: [out_119], Original ATen: [aten.convolution]
        buf441 = extern_kernels.convolution(buf440, primals_840, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf441, (4, 1088, 30, 30), (979200, 1, 32640, 1088))
        buf442 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [out_119, mul_29, out_120, out_121], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_41.run(buf442, primals_841, buf433, 3916800, grid=grid(3916800), stream=stream0)
        del primals_841
        # Topologically Sorted Source Nodes: [x_470], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_842, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (4, 256, 30, 30), (230400, 1, 7680, 256))
        buf444 = empty_strided_cuda((4, 256, 30, 30), (230400, 1, 7680, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_471, x_472], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_42.run(buf443, primals_843, primals_844, primals_845, primals_846, buf444, 921600, grid=grid(921600), stream=stream0)
        del primals_846
        # Topologically Sorted Source Nodes: [x_473], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, buf81, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (4, 384, 14, 14), (75264, 1, 5376, 384))
        # Topologically Sorted Source Nodes: [x_476], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf442, primals_852, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (4, 256, 30, 30), (230400, 1, 7680, 256))
        buf447 = empty_strided_cuda((4, 256, 30, 30), (230400, 1, 7680, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_477, x_478], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_42.run(buf446, primals_853, primals_854, primals_855, primals_856, buf447, 921600, grid=grid(921600), stream=stream0)
        del primals_856
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, buf82, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (4, 288, 14, 14), (56448, 1, 4032, 288))
        # Topologically Sorted Source Nodes: [x_482], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf442, primals_862, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf449, (4, 256, 30, 30), (230400, 1, 7680, 256))
        buf450 = empty_strided_cuda((4, 256, 30, 30), (230400, 1, 7680, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_483, x_484], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_42.run(buf449, primals_863, primals_864, primals_865, primals_866, buf450, 921600, grid=grid(921600), stream=stream0)
        del primals_866
        # Topologically Sorted Source Nodes: [x_485], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (4, 288, 30, 30), (259200, 1, 8640, 288))
        buf452 = empty_strided_cuda((4, 288, 30, 30), (259200, 1, 8640, 288), torch.float32)
        # Topologically Sorted Source Nodes: [x_486, x_487], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf451, primals_868, primals_869, primals_870, primals_871, buf452, 1036800, grid=grid(1036800), stream=stream0)
        del primals_871
        # Topologically Sorted Source Nodes: [x_488], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, buf84, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (4, 320, 14, 14), (62720, 1, 4480, 320))
        buf459 = empty_strided_cuda((4, 2080, 14, 14), (407680, 196, 14, 1), torch.float32)
        buf454 = reinterpret_tensor(buf459, (4, 1088, 14, 14), (407680, 196, 14, 1), 194432)  # alias
        buf455 = empty_strided_cuda((4, 1088, 14, 14), (213248, 1, 15232, 1088), torch.int8)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_44.run(buf442, buf454, buf455, 784, 1088, grid=grid(784, 1088), stream=stream0)
        buf456 = reinterpret_tensor(buf459, (4, 384, 14, 14), (407680, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_474, x_475], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf445, primals_848, primals_849, primals_850, primals_851, buf456, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf457 = reinterpret_tensor(buf459, (4, 288, 14, 14), (407680, 196, 14, 1), 75264)  # alias
        # Topologically Sorted Source Nodes: [x_480, x_481], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf448, primals_858, primals_859, primals_860, primals_861, buf457, 1152, 196, grid=grid(1152, 196), stream=stream0)
        buf458 = reinterpret_tensor(buf459, (4, 320, 14, 14), (407680, 196, 14, 1), 131712)  # alias
        # Topologically Sorted Source Nodes: [x_489, x_490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf453, primals_873, primals_874, primals_875, primals_876, buf458, 1280, 196, grid=grid(1280, 196), stream=stream0)
        buf460 = empty_strided_cuda((4, 2080, 14, 14), (407680, 1, 29120, 2080), torch.float32)
        # Topologically Sorted Source Nodes: [out_122], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf459, buf460, 8320, 196, grid=grid(8320, 196), stream=stream0)
        del buf454
        del buf456
        del buf457
        del buf458
        del buf459
        # Topologically Sorted Source Nodes: [x_491], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf460, primals_877, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf461, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_494], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf460, primals_882, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf463 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_495, x_496], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf462, primals_883, primals_884, primals_885, primals_886, buf463, 150528, grid=grid(150528), stream=stream0)
        del primals_886
        # Topologically Sorted Source Nodes: [x_497], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, buf85, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf465 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_498, x_499], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf464, primals_888, primals_889, primals_890, primals_891, buf465, 175616, grid=grid(175616), stream=stream0)
        del primals_891
        # Topologically Sorted Source Nodes: [x_500], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(buf465, buf86, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf467 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_123], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf461, primals_878, primals_879, primals_880, primals_881, buf466, primals_893, primals_894, primals_895, primals_896, buf467, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf468 = extern_kernels.convolution(buf467, primals_897, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf468, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf469 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [out_124, mul_30, out_125, out_126], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_52.run(buf469, primals_898, buf460, 1630720, grid=grid(1630720), stream=stream0)
        del primals_898
        # Topologically Sorted Source Nodes: [x_503], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, primals_899, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_506], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf469, primals_904, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf472 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_507, x_508], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf471, primals_905, primals_906, primals_907, primals_908, buf472, 150528, grid=grid(150528), stream=stream0)
        del primals_908
        # Topologically Sorted Source Nodes: [x_509], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, buf87, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf474 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_510, x_511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf473, primals_910, primals_911, primals_912, primals_913, buf474, 175616, grid=grid(175616), stream=stream0)
        del primals_913
        # Topologically Sorted Source Nodes: [x_512], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf474, buf88, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf476 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_127], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf470, primals_900, primals_901, primals_902, primals_903, buf475, primals_915, primals_916, primals_917, primals_918, buf476, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_128], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, primals_919, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf478 = buf477; del buf477  # reuse
        # Topologically Sorted Source Nodes: [out_128, mul_31, out_129, out_130], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_52.run(buf478, primals_920, buf469, 1630720, grid=grid(1630720), stream=stream0)
        del primals_920
        # Topologically Sorted Source Nodes: [x_515], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, primals_921, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_518], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf478, primals_926, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf481 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_519, x_520], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf480, primals_927, primals_928, primals_929, primals_930, buf481, 150528, grid=grid(150528), stream=stream0)
        del primals_930
        # Topologically Sorted Source Nodes: [x_521], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf481, buf89, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf482, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf483 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_522, x_523], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf482, primals_932, primals_933, primals_934, primals_935, buf483, 175616, grid=grid(175616), stream=stream0)
        del primals_935
        # Topologically Sorted Source Nodes: [x_524], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, buf90, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf485 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_131], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf479, primals_922, primals_923, primals_924, primals_925, buf484, primals_937, primals_938, primals_939, primals_940, buf485, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_132], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, primals_941, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf487 = buf486; del buf486  # reuse
        # Topologically Sorted Source Nodes: [out_132, mul_32, out_133, out_134], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_52.run(buf487, primals_942, buf478, 1630720, grid=grid(1630720), stream=stream0)
        del primals_942
        # Topologically Sorted Source Nodes: [x_527], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(buf487, primals_943, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_530], Original ATen: [aten.convolution]
        buf489 = extern_kernels.convolution(buf487, primals_948, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf489, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf490 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_531, x_532], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf489, primals_949, primals_950, primals_951, primals_952, buf490, 150528, grid=grid(150528), stream=stream0)
        del primals_952
        # Topologically Sorted Source Nodes: [x_533], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, buf91, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf492 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_534, x_535], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf491, primals_954, primals_955, primals_956, primals_957, buf492, 175616, grid=grid(175616), stream=stream0)
        del primals_957
        # Topologically Sorted Source Nodes: [x_536], Original ATen: [aten.convolution]
        buf493 = extern_kernels.convolution(buf492, buf92, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf493, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf494 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_135], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf488, primals_944, primals_945, primals_946, primals_947, buf493, primals_959, primals_960, primals_961, primals_962, buf494, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, primals_963, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf496 = buf495; del buf495  # reuse
        # Topologically Sorted Source Nodes: [out_136, mul_33, out_137, out_138], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_52.run(buf496, primals_964, buf487, 1630720, grid=grid(1630720), stream=stream0)
        del primals_964
        # Topologically Sorted Source Nodes: [x_539], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, primals_965, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_542], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf496, primals_970, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf499 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_543, x_544], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf498, primals_971, primals_972, primals_973, primals_974, buf499, 150528, grid=grid(150528), stream=stream0)
        del primals_974
        # Topologically Sorted Source Nodes: [x_545], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf499, buf93, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf501 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_546, x_547], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf500, primals_976, primals_977, primals_978, primals_979, buf501, 175616, grid=grid(175616), stream=stream0)
        del primals_979
        # Topologically Sorted Source Nodes: [x_548], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, buf94, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf503 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_139], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf497, primals_966, primals_967, primals_968, primals_969, buf502, primals_981, primals_982, primals_983, primals_984, buf503, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf504 = extern_kernels.convolution(buf503, primals_985, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf504, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf505 = buf504; del buf504  # reuse
        # Topologically Sorted Source Nodes: [out_140, mul_34, out_141, out_142], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_52.run(buf505, primals_986, buf496, 1630720, grid=grid(1630720), stream=stream0)
        del primals_986
        # Topologically Sorted Source Nodes: [x_551], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf505, primals_987, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_554], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf505, primals_992, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf508 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_555, x_556], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf507, primals_993, primals_994, primals_995, primals_996, buf508, 150528, grid=grid(150528), stream=stream0)
        del primals_996
        # Topologically Sorted Source Nodes: [x_557], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf508, buf95, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf510 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_558, x_559], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf509, primals_998, primals_999, primals_1000, primals_1001, buf510, 175616, grid=grid(175616), stream=stream0)
        del primals_1001
        # Topologically Sorted Source Nodes: [x_560], Original ATen: [aten.convolution]
        buf511 = extern_kernels.convolution(buf510, buf96, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf512 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_143], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf506, primals_988, primals_989, primals_990, primals_991, buf511, primals_1003, primals_1004, primals_1005, primals_1006, buf512, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_144], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf512, primals_1007, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf514 = buf513; del buf513  # reuse
        # Topologically Sorted Source Nodes: [out_144, mul_35, out_145, out_146], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_52.run(buf514, primals_1008, buf505, 1630720, grid=grid(1630720), stream=stream0)
        del primals_1008
        # Topologically Sorted Source Nodes: [x_563], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf514, primals_1009, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_566], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf514, primals_1014, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf517 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_567, x_568], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf516, primals_1015, primals_1016, primals_1017, primals_1018, buf517, 150528, grid=grid(150528), stream=stream0)
        del primals_1018
        # Topologically Sorted Source Nodes: [x_569], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(buf517, buf97, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf519 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_570, x_571], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf518, primals_1020, primals_1021, primals_1022, primals_1023, buf519, 175616, grid=grid(175616), stream=stream0)
        del primals_1023
        # Topologically Sorted Source Nodes: [x_572], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, buf98, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf521 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_147], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf515, primals_1010, primals_1011, primals_1012, primals_1013, buf520, primals_1025, primals_1026, primals_1027, primals_1028, buf521, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_148], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, primals_1029, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf523 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [out_148, mul_36, out_149, out_150], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_52.run(buf523, primals_1030, buf514, 1630720, grid=grid(1630720), stream=stream0)
        del primals_1030
        # Topologically Sorted Source Nodes: [x_575], Original ATen: [aten.convolution]
        buf524 = extern_kernels.convolution(buf523, primals_1031, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf524, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_578], Original ATen: [aten.convolution]
        buf525 = extern_kernels.convolution(buf523, primals_1036, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf525, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf526 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_579, x_580], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf525, primals_1037, primals_1038, primals_1039, primals_1040, buf526, 150528, grid=grid(150528), stream=stream0)
        del primals_1040
        # Topologically Sorted Source Nodes: [x_581], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, buf99, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf528 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_582, x_583], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf527, primals_1042, primals_1043, primals_1044, primals_1045, buf528, 175616, grid=grid(175616), stream=stream0)
        del primals_1045
        # Topologically Sorted Source Nodes: [x_584], Original ATen: [aten.convolution]
        buf529 = extern_kernels.convolution(buf528, buf100, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf529, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf530 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_151], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf524, primals_1032, primals_1033, primals_1034, primals_1035, buf529, primals_1047, primals_1048, primals_1049, primals_1050, buf530, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_152], Original ATen: [aten.convolution]
        buf531 = extern_kernels.convolution(buf530, primals_1051, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf531, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf532 = buf531; del buf531  # reuse
        # Topologically Sorted Source Nodes: [out_152, mul_37, out_153, out_154], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_52.run(buf532, primals_1052, buf523, 1630720, grid=grid(1630720), stream=stream0)
        del primals_1052
        # Topologically Sorted Source Nodes: [x_587], Original ATen: [aten.convolution]
        buf533 = extern_kernels.convolution(buf532, primals_1053, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf533, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_590], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf532, primals_1058, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf535 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_591, x_592], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf534, primals_1059, primals_1060, primals_1061, primals_1062, buf535, 150528, grid=grid(150528), stream=stream0)
        del primals_1062
        # Topologically Sorted Source Nodes: [x_593], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, buf101, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf537 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_594, x_595], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf536, primals_1064, primals_1065, primals_1066, primals_1067, buf537, 175616, grid=grid(175616), stream=stream0)
        del primals_1067
        # Topologically Sorted Source Nodes: [x_596], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, buf102, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf539 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_155], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf533, primals_1054, primals_1055, primals_1056, primals_1057, buf538, primals_1069, primals_1070, primals_1071, primals_1072, buf539, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, primals_1073, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf541 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [out_156, mul_38, out_157, out_158], Original ATen: [aten.convolution, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_relu_52.run(buf541, primals_1074, buf532, 1630720, grid=grid(1630720), stream=stream0)
        del primals_1074
        # Topologically Sorted Source Nodes: [x_599], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, primals_1075, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (4, 192, 14, 14), (37632, 1, 2688, 192))
        # Topologically Sorted Source Nodes: [x_602], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf541, primals_1080, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf544 = empty_strided_cuda((4, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_603, x_604], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf543, primals_1081, primals_1082, primals_1083, primals_1084, buf544, 150528, grid=grid(150528), stream=stream0)
        del primals_1084
        # Topologically Sorted Source Nodes: [x_605], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(buf544, buf103, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (4, 224, 14, 14), (43904, 1, 3136, 224))
        buf546 = empty_strided_cuda((4, 224, 14, 14), (43904, 1, 3136, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_606, x_607], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf545, primals_1086, primals_1087, primals_1088, primals_1089, buf546, 175616, grid=grid(175616), stream=stream0)
        del primals_1089
        # Topologically Sorted Source Nodes: [x_608], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(buf546, buf104, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (4, 256, 14, 14), (50176, 1, 3584, 256))
        buf548 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_159], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_51.run(buf542, primals_1076, primals_1077, primals_1078, primals_1079, buf547, primals_1091, primals_1092, primals_1093, primals_1094, buf548, 351232, grid=grid(351232), stream=stream0)
        # Topologically Sorted Source Nodes: [out_160], Original ATen: [aten.convolution]
        buf549 = extern_kernels.convolution(buf548, primals_1095, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf549, (4, 2080, 14, 14), (407680, 1, 29120, 2080))
        buf550 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [out_160, mul_39, out_161], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_53.run(buf550, primals_1096, buf541, 1630720, grid=grid(1630720), stream=stream0)
        del primals_1096
        # Topologically Sorted Source Nodes: [x_611], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_1097, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (4, 1536, 14, 14), (301056, 1, 21504, 1536))
        buf552 = empty_strided_cuda((4, 1536, 14, 14), (301056, 1, 21504, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [x_612, x_613], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_54.run(buf551, primals_1098, primals_1099, primals_1100, primals_1101, buf552, 1204224, grid=grid(1204224), stream=stream0)
        del primals_1101
        # Topologically Sorted Source Nodes: [x_614], Original ATen: [aten.avg_pool2d]
        buf553 = torch.ops.aten.avg_pool2d.default(buf552, [8, 8], [8, 8], [0, 0], False, False, None)
        buf554 = buf553
        del buf553
        buf555 = empty_strided_cuda((4, 1001), (1001, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_616], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1103, reinterpret_tensor(buf554, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_1102, (1536, 1001), (1, 1536), 0), alpha=1, beta=1, out=buf555)
        del primals_1103
    return (buf555, buf0, buf1, primals_3, primals_4, primals_5, buf2, primals_8, primals_9, primals_10, buf3, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, buf4, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, buf5, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, buf6, primals_48, primals_49, primals_50, buf7, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, buf8, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, buf9, primals_83, primals_84, primals_85, buf10, primals_88, primals_89, primals_90, primals_91, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, buf11, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, buf12, primals_115, primals_116, primals_117, buf13, primals_120, primals_121, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, buf14, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, buf15, primals_147, primals_148, primals_149, buf16, primals_152, primals_153, primals_154, primals_155, primals_156, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, buf17, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, buf18, primals_179, primals_180, primals_181, buf19, primals_184, primals_185, primals_186, primals_187, primals_188, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, buf20, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, buf21, primals_211, primals_212, primals_213, buf22, primals_216, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, buf23, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, buf24, primals_243, primals_244, primals_245, buf25, primals_248, primals_249, primals_250, primals_251, primals_252, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, buf26, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, buf27, primals_275, primals_276, primals_277, buf28, primals_280, primals_281, primals_282, primals_283, primals_284, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, buf29, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, buf30, primals_307, primals_308, primals_309, buf31, primals_312, primals_313, primals_314, primals_315, primals_316, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, buf32, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, buf33, primals_339, primals_340, primals_341, buf34, primals_344, primals_345, primals_346, primals_347, primals_348, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, buf35, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, buf36, primals_371, primals_372, primals_373, buf37, primals_376, primals_377, primals_378, primals_379, primals_380, buf38, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, buf39, primals_393, primals_394, primals_395, buf40, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, buf41, primals_413, primals_414, primals_415, buf42, primals_418, primals_419, primals_420, primals_421, primals_422, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, buf43, primals_435, primals_436, primals_437, buf44, primals_440, primals_441, primals_442, primals_443, primals_444, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, buf45, primals_457, primals_458, primals_459, buf46, primals_462, primals_463, primals_464, primals_465, primals_466, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, buf47, primals_479, primals_480, primals_481, buf48, primals_484, primals_485, primals_486, primals_487, primals_488, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, buf49, primals_501, primals_502, primals_503, buf50, primals_506, primals_507, primals_508, primals_509, primals_510, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, buf51, primals_523, primals_524, primals_525, buf52, primals_528, primals_529, primals_530, primals_531, primals_532, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, buf53, primals_545, primals_546, primals_547, buf54, primals_550, primals_551, primals_552, primals_553, primals_554, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, buf55, primals_567, primals_568, primals_569, buf56, primals_572, primals_573, primals_574, primals_575, primals_576, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, buf57, primals_589, primals_590, primals_591, buf58, primals_594, primals_595, primals_596, primals_597, primals_598, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, buf59, primals_611, primals_612, primals_613, buf60, primals_616, primals_617, primals_618, primals_619, primals_620, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, buf61, primals_633, primals_634, primals_635, buf62, primals_638, primals_639, primals_640, primals_641, primals_642, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, buf63, primals_655, primals_656, primals_657, buf64, primals_660, primals_661, primals_662, primals_663, primals_664, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, buf65, primals_677, primals_678, primals_679, buf66, primals_682, primals_683, primals_684, primals_685, primals_686, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, buf67, primals_699, primals_700, primals_701, buf68, primals_704, primals_705, primals_706, primals_707, primals_708, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, buf69, primals_721, primals_722, primals_723, buf70, primals_726, primals_727, primals_728, primals_729, primals_730, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, buf71, primals_743, primals_744, primals_745, buf72, primals_748, primals_749, primals_750, primals_751, primals_752, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, buf73, primals_765, primals_766, primals_767, buf74, primals_770, primals_771, primals_772, primals_773, primals_774, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, buf75, primals_787, primals_788, primals_789, buf76, primals_792, primals_793, primals_794, primals_795, primals_796, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, buf77, primals_809, primals_810, primals_811, buf78, primals_814, primals_815, primals_816, primals_817, primals_818, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, buf79, primals_831, primals_832, primals_833, buf80, primals_836, primals_837, primals_838, primals_839, primals_840, primals_842, primals_843, primals_844, primals_845, buf81, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, buf82, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, buf83, primals_868, primals_869, primals_870, buf84, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, buf85, primals_888, primals_889, primals_890, buf86, primals_893, primals_894, primals_895, primals_896, primals_897, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, buf87, primals_910, primals_911, primals_912, buf88, primals_915, primals_916, primals_917, primals_918, primals_919, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, buf89, primals_932, primals_933, primals_934, buf90, primals_937, primals_938, primals_939, primals_940, primals_941, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, buf91, primals_954, primals_955, primals_956, buf92, primals_959, primals_960, primals_961, primals_962, primals_963, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, buf93, primals_976, primals_977, primals_978, buf94, primals_981, primals_982, primals_983, primals_984, primals_985, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, buf95, primals_998, primals_999, primals_1000, buf96, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, buf97, primals_1020, primals_1021, primals_1022, buf98, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, buf99, primals_1042, primals_1043, primals_1044, buf100, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, buf101, primals_1064, primals_1065, primals_1066, buf102, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, buf103, primals_1086, primals_1087, primals_1088, buf104, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1097, primals_1098, primals_1099, primals_1100, buf105, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf128, buf129, buf130, buf131, buf132, buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf188, buf190, buf191, buf192, buf193, buf194, buf195, buf196, buf197, buf198, buf199, buf200, buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf209, buf210, buf211, buf212, buf214, buf215, buf216, buf217, buf218, buf219, buf220, buf221, buf222, buf223, buf224, buf226, buf227, buf228, buf229, buf230, buf231, buf232, buf233, buf234, buf235, buf236, buf238, buf239, buf240, buf241, buf242, buf243, buf244, buf245, buf246, buf247, buf248, buf250, buf251, buf252, buf253, buf254, buf255, buf256, buf258, buf262, buf263, buf264, buf265, buf266, buf267, buf268, buf269, buf271, buf272, buf273, buf274, buf275, buf276, buf277, buf278, buf280, buf281, buf282, buf283, buf284, buf285, buf286, buf287, buf289, buf290, buf291, buf292, buf293, buf294, buf295, buf296, buf298, buf299, buf300, buf301, buf302, buf303, buf304, buf305, buf307, buf308, buf309, buf310, buf311, buf312, buf313, buf314, buf316, buf317, buf318, buf319, buf320, buf321, buf322, buf323, buf325, buf326, buf327, buf328, buf329, buf330, buf331, buf332, buf334, buf335, buf336, buf337, buf338, buf339, buf340, buf341, buf343, buf344, buf345, buf346, buf347, buf348, buf349, buf350, buf352, buf353, buf354, buf355, buf356, buf357, buf358, buf359, buf361, buf362, buf363, buf364, buf365, buf366, buf367, buf368, buf370, buf371, buf372, buf373, buf374, buf375, buf376, buf377, buf379, buf380, buf381, buf382, buf383, buf384, buf385, buf386, buf388, buf389, buf390, buf391, buf392, buf393, buf394, buf395, buf397, buf398, buf399, buf400, buf401, buf402, buf403, buf404, buf406, buf407, buf408, buf409, buf410, buf411, buf412, buf413, buf415, buf416, buf417, buf418, buf419, buf420, buf421, buf422, buf424, buf425, buf426, buf427, buf428, buf429, buf430, buf431, buf433, buf434, buf435, buf436, buf437, buf438, buf439, buf440, buf442, buf443, buf444, buf445, buf446, buf447, buf448, buf449, buf450, buf451, buf452, buf453, buf455, buf460, buf461, buf462, buf463, buf464, buf465, buf466, buf467, buf469, buf470, buf471, buf472, buf473, buf474, buf475, buf476, buf478, buf479, buf480, buf481, buf482, buf483, buf484, buf485, buf487, buf488, buf489, buf490, buf491, buf492, buf493, buf494, buf496, buf497, buf498, buf499, buf500, buf501, buf502, buf503, buf505, buf506, buf507, buf508, buf509, buf510, buf511, buf512, buf514, buf515, buf516, buf517, buf518, buf519, buf520, buf521, buf523, buf524, buf525, buf526, buf527, buf528, buf529, buf530, buf532, buf533, buf534, buf535, buf536, buf537, buf538, buf539, buf541, buf542, buf543, buf544, buf545, buf546, buf547, buf548, buf550, buf551, buf552, reinterpret_tensor(buf554, (4, 1536), (1536, 1), 0), primals_1102, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 512, 512), (786432, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
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
    primals_27 = rand_strided((96, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_57 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((32, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((48, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((64, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((320, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((384, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((256, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((384, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((192, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((128, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((160, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((1088, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((256, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((384, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((256, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((288, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((256, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((288, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((320, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_924 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_927 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_930 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_933 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_938 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_939 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_940 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_941 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_942 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_943 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_944 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_945 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_946 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_947 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_948 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_949 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_950 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_951 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_952 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_953 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_954 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_955 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_956 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_957 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_958 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_959 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_960 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_961 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_962 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_963 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_964 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_965 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_966 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_967 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_968 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_969 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_970 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_971 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_972 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_973 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_974 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_975 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_976 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_977 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_978 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_979 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_980 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_981 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_982 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_983 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_984 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_985 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_986 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_987 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_988 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_989 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_990 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_991 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_992 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_993 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_994 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_995 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_996 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_997 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_998 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_999 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1000 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1001 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1002 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1003 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1004 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1005 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1006 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1007 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1008 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1009 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1010 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1011 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1012 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1013 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1014 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1015 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1016 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1017 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1018 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1019 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1020 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1021 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1022 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1023 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1024 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1025 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1026 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1027 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1028 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1029 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1030 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1031 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1032 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1033 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1034 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1035 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1036 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1037 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1038 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1039 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1040 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1041 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1042 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1043 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1044 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1045 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1046 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1047 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1048 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1049 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1050 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1051 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1052 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1053 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1054 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1055 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1056 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1057 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1058 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1059 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1060 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1061 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1062 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1063 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1064 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1065 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1066 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1067 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1068 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1069 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1070 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1071 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1072 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1073 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1074 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1075 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1076 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1077 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1078 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1079 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1080 = rand_strided((192, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1081 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1082 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1083 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1084 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1085 = rand_strided((224, 192, 1, 3), (576, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1086 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1087 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1088 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1089 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1090 = rand_strided((256, 224, 3, 1), (672, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1091 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1092 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1093 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1094 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1095 = rand_strided((2080, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1096 = rand_strided((2080, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1097 = rand_strided((1536, 2080, 1, 1), (2080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1098 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1099 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1100 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1101 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1102 = rand_strided((1001, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_1103 = rand_strided((1001, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
