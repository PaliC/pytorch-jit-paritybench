# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/im/cimtz22r7cnmkx74rlnxkzwf2mja43c24y422bm3tiez7rdhzrwy.py
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
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/xu/cxuyr4fep26d5mfh7qkozfo6556nflxkho5wmw63uvfxrps7idqh.py
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
    size_hints={'y': 16384, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/p7/cp7tprkttzn6c4zelt4zlk2e64rxsdchb3h7sj56aubsjhrbaddq.py
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


# kernel path: inductor_cache/vi/cvie4kd26wwgmz4g7wrpzie7o4okvzomikujqvb4lto6lp65qgzq.py
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


# kernel path: inductor_cache/ik/ciknwdrsy4sriuk3gkoyca3zk7froufd6th76n5bqej4aitv54hq.py
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
    size_hints={'y': 32768, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/yj/cyjkqk4vlg6jzg42qfwi6wyvdjbewrum2nlnpsrmy4nvrgaxmwnr.py
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
    size_hints={'y': 65536, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/cj/ccj2u625dyrdvlhmntns3ykgsaxnkcneucfyqhazy77qjzxiyd7s.py
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


# kernel path: inductor_cache/yh/cyheq2w6irl3izitmg5zrugufoymkrojdjp4k225nqqpx44x5ioz.py
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/lv/clvxctlgzuncwy6aizpbmkanjrey7fjhsnmjswv6duaklsmvdcxq.py
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
    size_hints={'y': 262144, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/hj/chj47qmbbyfmvpp5zuabjojceccoqdvlofuskl5mpdbuj4ltv7do.py
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
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/3p/c3p2whbx7kmzwq7lwdnrouck45mqjegazf4d6g222nw3ne25hzzq.py
# Topologically Sorted Source Nodes: [input_2, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   x => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/2u/c2ucq5hn53x3peyn7aaweyhyrxn74qzudm662vgltkgboau27bf5.py
# Topologically Sorted Source Nodes: [input_4, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_3, mul_4, mul_5, sub_1
#   x_1 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
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


# kernel path: inductor_cache/ga/cgaqyfdxfq42fpkbi57pels3bxxey7hieflcvj3s6b434js3kylr.py
# Topologically Sorted Source Nodes: [input_6, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_5, mul_7, mul_8, sub_2
#   x_2 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/kk/ckk62pcjfwf4odkkjtuz2x3sljj7fzikrlmcfzva52tyberbla5e.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_3 => getitem, getitem_1
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
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_21(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/yw/cywjefjfqksqyztfttrxqma6lgfvahj3j2xmwgctvn3l3eez2pdj.py
# Topologically Sorted Source Nodes: [input_8, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_8 => add_7, mul_10, mul_11, sub_3
#   x_4 => relu_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ou/couwbl44km757lgqlizokxw63cnze4zcd2igyx43i37phx5e4vub.py
# Topologically Sorted Source Nodes: [input_10, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_9, mul_13, mul_14, sub_4
#   x_5 => relu_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/i3/ci3i4ssplgyvom3mhre6vacdz655xk6xpf4qovn424n7souvihli.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_6 => getitem_2, getitem_3
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
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_24(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/5t/c5tobb7xyzc3s3ziuf5jycmynahjvbwu27hrpfogfe2muvfyo6md.py
# Topologically Sorted Source Nodes: [input_14, branch5x5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch5x5 => relu_6
#   input_14 => add_13, mul_19, mul_20, sub_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/4s/c4sbsvivec3vpvzb3vne4bmte2njyhnp4tw3ha65ybqn72uw72u6.py
# Topologically Sorted Source Nodes: [input_18, branch3x3dbl], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl => relu_8
#   input_18 => add_17, mul_25, mul_26, sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
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


# kernel path: inductor_cache/b4/cb4fqnvoblke5mj4ng7wv76kog2x4ndagt3qulap7ohnfikgza6t.py
# Topologically Sorted Source Nodes: [input_20, branch3x3dbl_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl_1 => relu_9
#   input_20 => add_19, mul_28, mul_29, sub_9
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/s2/cs2iqrm7aiv4lt3r6pfhutpimomahdjg6g4ecu6f2rqi37t3ixie.py
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
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((62) * ((62) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (62)))*((62) * ((62) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (62))) + ((-1)*x1*((62) * ((62) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (62)))) + ((-1)*x2*((62) * ((62) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (62)))) + ((62) * ((62) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (62))) + ((62) * ((62) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (62)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fl/cflp2xvkloa4y5or555uq4qdpf4fvqanrplx3gdwguh2rfsbhzqc.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_7 => cat
# Graph fragment:
#   %cat : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_5, %relu_7, %relu_10, %relu_11], 1), kwargs = {})
triton_poi_fused_cat_29 = async_compile.triton('triton_poi_fused_cat_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3810304
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


# kernel path: inductor_cache/zh/czhuotxt74mhcqabz4qluqmmdd2ptc4tqavwle7ukputi5gudr6d.py
# Topologically Sorted Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_2 => avg_pool2d_1
# Graph fragment:
#   %avg_pool2d_1 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_30 = async_compile.triton('triton_poi_fused_avg_pool2d_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3810304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 15616) % 61)
    x1 = ((xindex // 256) % 61)
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
    tmp11 = tl.load(in_ptr0 + ((-15872) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-15616) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-15360) + x6), tmp23 & xmask, other=0.0)
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
    tmp44 = tl.load(in_ptr0 + (15360 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (15616 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (15872 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((62) * ((62) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (62)))*((62) * ((62) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (62))) + ((-1)*x1*((62) * ((62) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (62)))) + ((-1)*x2*((62) * ((62) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (62)))) + ((62) * ((62) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (62))) + ((62) * ((62) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (62)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4u/c4uraykn3tkpinhqpbeadkbj5zm42wrojuirt7wfssmdbazczgmn.py
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_8 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %relu_14, %relu_17, %relu_18], 1), kwargs = {})
triton_poi_fused_cat_31 = async_compile.triton('triton_poi_fused_cat_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4286592
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


# kernel path: inductor_cache/bt/cbt2hrjuyrh3kt5gzz6sl6kxwmxdo3iaq2pa52fikhdnsmpyz3zf.py
# Topologically Sorted Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_4 => avg_pool2d_2
# Graph fragment:
#   %avg_pool2d_2 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_1, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_32 = async_compile.triton('triton_poi_fused_avg_pool2d_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4286592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 17568) % 61)
    x1 = ((xindex // 288) % 61)
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
    tmp11 = tl.load(in_ptr0 + ((-17856) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-17568) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-17280) + x6), tmp23 & xmask, other=0.0)
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
    tmp44 = tl.load(in_ptr0 + (17280 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (17568 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (17856 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((62) * ((62) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (62)))*((62) * ((62) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (62))) + ((-1)*x1*((62) * ((62) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (62)))) + ((-1)*x2*((62) * ((62) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (62)))) + ((62) * ((62) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (62))) + ((62) * ((62) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (62)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/33/c337gtzu6xt36aba6hdclbjqzym26dhizuorw2mgze5rz2rdszen.py
# Topologically Sorted Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   branch_pool_6 => _low_memory_max_pool2d_with_offsets_2, getitem_5
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_2, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
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
    size_hints={'y': 4096, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_33(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3600
    xnumel = 288
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
    tmp0 = tl.load(in_ptr0 + (x3 + 576*y0 + 35136*y1 + 1071648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (288 + x3 + 576*y0 + 35136*y1 + 1071648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (576 + x3 + 576*y0 + 35136*y1 + 1071648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17568 + x3 + 576*y0 + 35136*y1 + 1071648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (17856 + x3 + 576*y0 + 35136*y1 + 1071648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (18144 + x3 + 576*y0 + 35136*y1 + 1071648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (35136 + x3 + 576*y0 + 35136*y1 + 1071648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (35424 + x3 + 576*y0 + 35136*y1 + 1071648*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (35712 + x3 + 576*y0 + 35136*y1 + 1071648*y2), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y4 + 900*x3 + 691200*y2), tmp16, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 288*y5), tmp41, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/vf/cvfva5jstwixefrccr3at3qqxrwfsoybzia7jh7sedgpl4ipxndr.py
# Topologically Sorted Source Nodes: [input_54, branch3x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3 => relu_26
#   input_54 => add_53, mul_79, mul_80, sub_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %relu_26 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2 + 900*y0 + 691200*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/p4/cp4yk753sv77w2geugtptfe2k7rbtph2ule3zdn72od7ic3k2jud.py
# Topologically Sorted Source Nodes: [input_60, branch3x3dbl_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl_11 => relu_29
#   input_60 => add_59, mul_88, mul_89, sub_29
# Graph fragment:
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_237), kwargs = {})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_239), kwargs = {})
#   %relu_29 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_59,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 900
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 86400*y1), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 900*y0 + 691200*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/io/cio6onfqzl6onw3zaoeqwxnjnr52srr7egv52segf25dmnjcj4op.py
# Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_10 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_26, %relu_29, %getitem_4], 1), kwargs = {})
triton_poi_fused_cat_36 = async_compile.triton('triton_poi_fused_cat_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 900
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
    tmp0 = tl.load(in_ptr0 + (x2 + 900*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 768*x2 + 691200*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jl/cjlrm5visvzxjreqbmuxw5nupjpn4i6djbi6yrn7zc5zet6q2lqx.py
# Topologically Sorted Source Nodes: [input_64, branch7x7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7 => relu_31
#   input_64 => add_63, mul_94, mul_95, sub_31
# Graph fragment:
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_249), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %unsqueeze_253), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %unsqueeze_255), kwargs = {})
#   %relu_31 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_63,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/jk/cjkwwqkhcaohe64krvhuy3fdb4it2uxrpiyo6ldtpisnjqr4aiyd.py
# Topologically Sorted Source Nodes: [branch_pool_7], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_7 => avg_pool2d_3
# Graph fragment:
#   %avg_pool2d_3 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_3, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_38 = async_compile.triton('triton_poi_fused_avg_pool2d_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2764800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 23040) % 30)
    x1 = ((xindex // 768) % 30)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-23808) + x6), tmp10, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-23040) + x6), tmp16, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-22272) + x6), tmp23, other=0.0)
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
    tmp44 = tl.load(in_ptr0 + (22272 + x6), tmp43, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (23040 + x6), tmp46, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (23808 + x6), tmp49, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((31) * ((31) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (31)))*((31) * ((31) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (31))) + ((-1)*x1*((31) * ((31) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (31)))) + ((-1)*x2*((31) * ((31) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (31)))) + ((31) * ((31) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (31))) + ((31) * ((31) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (31)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/g4/cg4lsevhbcqolbaswi3jqru4ebau2rl5svihgyz6jctjbo554jkq.py
# Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_11 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_30, %relu_33, %relu_38, %relu_39], 1), kwargs = {})
triton_poi_fused_cat_39 = async_compile.triton('triton_poi_fused_cat_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2764800
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


# kernel path: inductor_cache/7z/c7z7326jsxmansv5ilvc7dt5dygtfhwpjotihtfwzxxeb4gkyku4.py
# Topologically Sorted Source Nodes: [input_84, branch7x7_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7_3 => relu_41
#   input_84 => add_83, mul_124, mul_125, sub_41
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_333), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_83,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/jy/cjylggep5vtiuplqu2i4okjf4q33ky6crg4ls2qnt4jyni2ngevl.py
# Topologically Sorted Source Nodes: [input_124, branch7x7_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7_9 => relu_61
#   input_124 => add_123, mul_184, mul_185, sub_61
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_123,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 691200
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


# kernel path: inductor_cache/7g/c7gksze6jiscj4ekra2t2i52wif65iwokw6idduocrsys55uqcmf.py
# Topologically Sorted Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   branch_pool_15 => _low_memory_max_pool2d_with_offsets_3, getitem_7
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
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
    size_hints={'y': 1024, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_42(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x3 + 1536*y0 + 46080*y1 + 691200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (768 + x3 + 1536*y0 + 46080*y1 + 691200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1536 + x3 + 1536*y0 + 46080*y1 + 691200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (23040 + x3 + 1536*y0 + 46080*y1 + 691200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (23808 + x3 + 1536*y0 + 46080*y1 + 691200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (24576 + x3 + 1536*y0 + 46080*y1 + 691200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (46080 + x3 + 1536*y0 + 46080*y1 + 691200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (46848 + x3 + 1536*y0 + 46080*y1 + 691200*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (47616 + x3 + 1536*y0 + 46080*y1 + 691200*y2), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y4 + 196*x3 + 250880*y2), tmp16, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 768*y5), tmp41, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbr4szrkb77vxzzmywaencbdfxeb7lr2qdespjzqz5nub7bkdixl.py
# Topologically Sorted Source Nodes: [input_144, branch3x3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3_2 => relu_71
#   input_144 => add_143, mul_214, mul_215, sub_71
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_569), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_214, %unsqueeze_573), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, %unsqueeze_575), kwargs = {})
#   %relu_71 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_143,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2 + 196*y0 + 250880*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/q3/cq347fm5gsorf4vdlkzcepmtxab5pjjsa4n7meilckaua4lvht57.py
# Topologically Sorted Source Nodes: [input_152, branch7x7x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7x3_3 => relu_75
#   input_152 => add_151, mul_226, mul_227, sub_75
# Graph fragment:
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_75, %unsqueeze_601), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %unsqueeze_603), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %unsqueeze_605), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %unsqueeze_607), kwargs = {})
#   %relu_75 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_151,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 37632*y1), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 196*y0 + 250880*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/rf/crfvftvmlycdn7zhdbb7wlrtbtd7v6fot64wwbvtvo7qjskpdgsr.py
# Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_15 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_71, %relu_75, %getitem_6], 1), kwargs = {})
triton_poi_fused_cat_45 = async_compile.triton('triton_poi_fused_cat_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_45(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + 196*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1280*x2 + 250880*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wb/cwbonpvsi53npzc52dodiooa527gp4agnjfl5nw4njqcyetah65e.py
# Topologically Sorted Source Nodes: [input_156, branch3x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3_3 => relu_77
#   input_156 => add_155, mul_232, mul_233, sub_77
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_617), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_232, %unsqueeze_621), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_233, %unsqueeze_623), kwargs = {})
#   %relu_77 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
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


# kernel path: inductor_cache/2f/c2f4tkw7ts2b6vrbc54do4jex325drxeb6wa4fgo6mehmexgv6sq.py
# Topologically Sorted Source Nodes: [branch3x3_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   branch3x3_4 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_78, %relu_79], 1), kwargs = {})
triton_poi_fused_cat_47 = async_compile.triton('triton_poi_fused_cat_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 196) % 768)
    x0 = (xindex % 196)
    x2 = xindex // 150528
    x3 = (xindex % 150528)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (384*x0 + 75264*x2 + (x1)), tmp4, eviction_policy='evict_last', other=0.0)
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
    tmp28 = tl.load(in_ptr5 + (384*x0 + 75264*x2 + ((-384) + x1)), tmp25, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x3 + 401408*x2), tmp48, None)
''', device_str='cuda')


# kernel path: inductor_cache/h7/ch7ebrjb45aetugijv5772qj2otyofyyyh47pi32622vt4xxbe74.py
# Topologically Sorted Source Nodes: [input_162, branch3x3dbl_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl_12 => relu_80
#   input_162 => add_161, mul_241, mul_242, sub_80
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_641), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_241, %unsqueeze_645), kwargs = {})
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %unsqueeze_647), kwargs = {})
#   %relu_80 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_161,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
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


# kernel path: inductor_cache/mg/cmgxu4skrbq6h23qzkyg4c2c4k4dptt4yexfxkmw2dndxe4vsmpg.py
# Topologically Sorted Source Nodes: [branch_pool_16], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_16 => avg_pool2d_7
# Graph fragment:
#   %avg_pool2d_7 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_8, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_49 = async_compile.triton('triton_poi_fused_avg_pool2d_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 17920) % 14)
    x1 = ((xindex // 1280) % 14)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-19200) + x6), tmp10, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-17920) + x6), tmp16, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-16640) + x6), tmp23, other=0.0)
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
    tmp44 = tl.load(in_ptr0 + (16640 + x6), tmp43, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (17920 + x6), tmp46, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (19200 + x6), tmp49, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((15) * ((15) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (15)))*((15) * ((15) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (15))) + ((-1)*x1*((15) * ((15) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (15)))) + ((-1)*x2*((15) * ((15) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (15)))) + ((15) * ((15) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (15))) + ((15) * ((15) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (15)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/gn/cgnp5srrkiqno3tteni5kvqj3ubcayo2s3ckg5x3sej7kcwhgswb.py
# Topologically Sorted Source Nodes: [input_154, branch1x1_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch1x1_7 => relu_76
#   input_154 => add_153, mul_229, mul_230, sub_76
# Graph fragment:
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_609), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_611), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_229, %unsqueeze_613), kwargs = {})
#   %add_153 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_230, %unsqueeze_615), kwargs = {})
#   %relu_76 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_153,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2 + 196*y0 + 401408*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/7g/c7goakvjeiftdy7ypxbuakz4b5ydbdjd6yipg5ekgxas64lln3uw.py
# Topologically Sorted Source Nodes: [input_170, branch_pool_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch_pool_17 => relu_84
#   input_170 => add_169, mul_253, mul_254, sub_84
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_673), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_253, %unsqueeze_677), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_254, %unsqueeze_679), kwargs = {})
#   %relu_84 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_169,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 37632*y1), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 196*y0 + 401408*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/eu/ceusjripnxquqtp5j36lpiibfdoswby4pgwsqk54giu727io63hp.py
# Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_16 => cat_11
# Graph fragment:
#   %cat_11 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_76, %cat_9, %cat_10, %relu_84], 1), kwargs = {})
triton_poi_fused_cat_52 = async_compile.triton('triton_poi_fused_cat_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_52(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + 196*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 2048*x2 + 401408*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lx/clxhkkdxznpa62pckox3ujmv3eamt63eoe4ku6l4uswa2mtjqab5.py
# Topologically Sorted Source Nodes: [branch_pool_18], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_18 => avg_pool2d_8
# Graph fragment:
#   %avg_pool2d_8 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_11, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_53 = async_compile.triton('triton_poi_fused_avg_pool2d_53', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 28672) % 14)
    x1 = ((xindex // 2048) % 14)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-30720) + x6), tmp10, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-28672) + x6), tmp16, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-26624) + x6), tmp23, other=0.0)
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
    tmp44 = tl.load(in_ptr0 + (26624 + x6), tmp43, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (28672 + x6), tmp46, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (30720 + x6), tmp49, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((15) * ((15) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (15)))*((15) * ((15) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (15))) + ((-1)*x1*((15) * ((15) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (15)))) + ((-1)*x2*((15) * ((15) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (15)))) + ((15) * ((15) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (15))) + ((15) * ((15) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (15)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473 = args
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
    assert_size_stride(primals_472, (1000, 2048), (2048, 1))
    assert_size_stride(primals_473, (1000, ), (1, ))
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
        buf8 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_72, buf8, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_72
        buf9 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_82, buf9, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_82
        buf10 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_87, buf10, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_87
        buf11 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_107, buf11, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_107
        buf12 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_117, buf12, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_117
        buf13 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_122, buf13, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_122
        buf14 = empty_strided_cuda((384, 288, 3, 3), (2592, 1, 864, 288), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_132, buf14, 110592, 9, grid=grid(110592, 9), stream=stream0)
        del primals_132
        buf15 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_142, buf15, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_142
        buf16 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_147, buf16, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_147
        buf17 = empty_strided_cuda((128, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_162, buf17, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_162
        buf18 = empty_strided_cuda((192, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_167, buf18, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del primals_167
        buf19 = empty_strided_cuda((128, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_177, buf19, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_177
        buf20 = empty_strided_cuda((128, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_182, buf20, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_182
        buf21 = empty_strided_cuda((128, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_187, buf21, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_187
        buf22 = empty_strided_cuda((192, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_192, buf22, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del primals_192
        buf23 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_212, buf23, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_212
        buf24 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_217, buf24, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_217
        buf25 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_227, buf25, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_227
        buf26 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_232, buf26, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_232
        buf27 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_237, buf27, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_237
        buf28 = empty_strided_cuda((192, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_242, buf28, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_242
        buf29 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_262, buf29, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_262
        buf30 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_267, buf30, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_267
        buf31 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_277, buf31, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_277
        buf32 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_282, buf32, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_282
        buf33 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_287, buf33, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_287
        buf34 = empty_strided_cuda((192, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_292, buf34, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_292
        buf35 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_312, buf35, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_312
        buf36 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_317, buf36, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_317
        buf37 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_327, buf37, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_327
        buf38 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_332, buf38, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_332
        buf39 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_337, buf39, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_337
        buf40 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_342, buf40, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_342
        buf41 = empty_strided_cuda((320, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_357, buf41, 61440, 9, grid=grid(61440, 9), stream=stream0)
        del primals_357
        buf42 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_367, buf42, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_367
        buf43 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_372, buf43, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_372
        buf44 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_377, buf44, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_377
        buf45 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_392, buf45, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_392
        buf46 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_397, buf46, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_397
        buf47 = empty_strided_cuda((384, 448, 3, 3), (4032, 1, 1344, 448), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_407, buf47, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del primals_407
        buf48 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_412, buf48, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_412
        buf49 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_417, buf49, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_417
        buf50 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_437, buf50, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_437
        buf51 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_442, buf51, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_442
        buf52 = empty_strided_cuda((384, 448, 3, 3), (4032, 1, 1344, 448), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_452, buf52, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del primals_452
        buf53 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_457, buf53, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_457
        buf54 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_462, buf54, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_462
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 32, 255, 255), (2080800, 1, 8160, 32))
        buf56 = empty_strided_cuda((4, 32, 255, 255), (2080800, 1, 8160, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf55, primals_3, primals_4, primals_5, primals_6, buf56, 8323200, grid=grid(8323200), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, buf2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 32, 253, 253), (2048288, 1, 8096, 32))
        buf58 = empty_strided_cuda((4, 32, 253, 253), (2048288, 1, 8096, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf57, primals_8, primals_9, primals_10, primals_11, buf58, 8193152, grid=grid(8193152), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 64, 253, 253), (4096576, 1, 16192, 64))
        buf60 = empty_strided_cuda((4, 64, 253, 253), (4096576, 1, 16192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf59, primals_13, primals_14, primals_15, primals_16, buf60, 16386304, grid=grid(16386304), stream=stream0)
        del primals_16
        buf61 = empty_strided_cuda((4, 64, 126, 126), (1016064, 1, 8064, 64), torch.float32)
        buf62 = empty_strided_cuda((4, 64, 126, 126), (1016064, 1, 8064, 64), torch.int8)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_21.run(buf60, buf61, buf62, 4064256, grid=grid(4064256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf61, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 80, 126, 126), (1270080, 1, 10080, 80))
        buf64 = empty_strided_cuda((4, 80, 126, 126), (1270080, 1, 10080, 80), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf63, primals_18, primals_19, primals_20, primals_21, buf64, 5080320, grid=grid(5080320), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, buf4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 192, 124, 124), (2952192, 1, 23808, 192))
        buf66 = empty_strided_cuda((4, 192, 124, 124), (2952192, 1, 23808, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf65, primals_23, primals_24, primals_25, primals_26, buf66, 11808768, grid=grid(11808768), stream=stream0)
        del primals_26
        buf67 = empty_strided_cuda((4, 192, 61, 61), (714432, 1, 11712, 192), torch.float32)
        buf68 = empty_strided_cuda((4, 192, 61, 61), (714432, 1, 11712, 192), torch.int8)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_24.run(buf66, buf67, buf68, 2857728, grid=grid(2857728), stream=stream0)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf67, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 64, 61, 61), (238144, 1, 3904, 64))
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf67, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf71 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, branch5x5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf70, primals_33, primals_34, primals_35, primals_36, buf71, 714432, grid=grid(714432), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, buf5, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 64, 61, 61), (238144, 1, 3904, 64))
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf67, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf74 = empty_strided_cuda((4, 64, 61, 61), (238144, 1, 3904, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, branch3x3dbl], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf73, primals_43, primals_44, primals_45, primals_46, buf74, 952576, grid=grid(952576), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 96, 61, 61), (357216, 1, 5856, 96))
        buf76 = empty_strided_cuda((4, 96, 61, 61), (357216, 1, 5856, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_20, branch3x3dbl_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf75, primals_48, primals_49, primals_50, primals_51, buf76, 1428864, grid=grid(1428864), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 96, 61, 61), (357216, 1, 5856, 96))
        buf78 = empty_strided_cuda((4, 192, 61, 61), (714432, 1, 11712, 192), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_28.run(buf67, buf78, 2857728, grid=grid(2857728), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 32, 61, 61), (119072, 1, 1952, 32))
        buf80 = empty_strided_cuda((4, 256, 61, 61), (952576, 1, 15616, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_29.run(buf69, primals_28, primals_29, primals_30, primals_31, buf72, primals_38, primals_39, primals_40, primals_41, buf77, primals_53, primals_54, primals_55, primals_56, buf79, primals_58, primals_59, primals_60, primals_61, buf80, 3810304, grid=grid(3810304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 64, 61, 61), (238144, 1, 3904, 64))
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf80, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf83 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, branch5x5_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf82, primals_68, primals_69, primals_70, primals_71, buf83, 714432, grid=grid(714432), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, buf8, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 64, 61, 61), (238144, 1, 3904, 64))
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf80, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf86 = empty_strided_cuda((4, 64, 61, 61), (238144, 1, 3904, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_32, branch3x3dbl_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf85, primals_78, primals_79, primals_80, primals_81, buf86, 952576, grid=grid(952576), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 96, 61, 61), (357216, 1, 5856, 96))
        buf88 = empty_strided_cuda((4, 96, 61, 61), (357216, 1, 5856, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, branch3x3dbl_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf87, primals_83, primals_84, primals_85, primals_86, buf88, 1428864, grid=grid(1428864), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 96, 61, 61), (357216, 1, 5856, 96))
        buf90 = empty_strided_cuda((4, 256, 61, 61), (952576, 1, 15616, 256), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_30.run(buf80, buf90, 3810304, grid=grid(3810304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf92 = empty_strided_cuda((4, 288, 61, 61), (1071648, 1, 17568, 288), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_31.run(buf81, primals_63, primals_64, primals_65, primals_66, buf84, primals_73, primals_74, primals_75, primals_76, buf89, primals_88, primals_89, primals_90, primals_91, buf91, primals_93, primals_94, primals_95, primals_96, buf92, 4286592, grid=grid(4286592), stream=stream0)
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 64, 61, 61), (238144, 1, 3904, 64))
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf92, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 48, 61, 61), (178608, 1, 2928, 48))
        buf95 = empty_strided_cuda((4, 48, 61, 61), (178608, 1, 2928, 48), torch.float32)
        # Topologically Sorted Source Nodes: [input_42, branch5x5_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf94, primals_103, primals_104, primals_105, primals_106, buf95, 714432, grid=grid(714432), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, buf11, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 64, 61, 61), (238144, 1, 3904, 64))
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf92, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf98 = empty_strided_cuda((4, 64, 61, 61), (238144, 1, 3904, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_46, branch3x3dbl_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf97, primals_113, primals_114, primals_115, primals_116, buf98, 952576, grid=grid(952576), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 96, 61, 61), (357216, 1, 5856, 96))
        buf100 = empty_strided_cuda((4, 96, 61, 61), (357216, 1, 5856, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_48, branch3x3dbl_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf99, primals_118, primals_119, primals_120, primals_121, buf100, 1428864, grid=grid(1428864), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 96, 61, 61), (357216, 1, 5856, 96))
        buf102 = empty_strided_cuda((4, 288, 61, 61), (1071648, 1, 17568, 288), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_32.run(buf92, buf102, 4286592, grid=grid(4286592), stream=stream0)
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf104 = empty_strided_cuda((4, 288, 61, 61), (1071648, 1, 17568, 288), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_31.run(buf93, primals_98, primals_99, primals_100, primals_101, buf96, primals_108, primals_109, primals_110, primals_111, buf101, primals_123, primals_124, primals_125, primals_126, buf103, primals_128, primals_129, primals_130, primals_131, buf104, 4286592, grid=grid(4286592), stream=stream0)
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, buf14, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 384, 30, 30), (345600, 1, 11520, 384))
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf104, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 64, 61, 61), (238144, 1, 3904, 64))
        buf107 = empty_strided_cuda((4, 64, 61, 61), (238144, 1, 3904, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_56, branch3x3dbl_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf106, primals_138, primals_139, primals_140, primals_141, buf107, 952576, grid=grid(952576), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 96, 61, 61), (357216, 1, 5856, 96))
        buf109 = empty_strided_cuda((4, 96, 61, 61), (357216, 1, 5856, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_58, branch3x3dbl_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf108, primals_143, primals_144, primals_145, primals_146, buf109, 1428864, grid=grid(1428864), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, buf16, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 96, 30, 30), (86400, 1, 2880, 96))
        buf115 = empty_strided_cuda((4, 768, 30, 30), (691200, 900, 30, 1), torch.float32)
        buf111 = reinterpret_tensor(buf115, (4, 288, 30, 30), (691200, 900, 30, 1), 432000)  # alias
        buf112 = empty_strided_cuda((4, 288, 30, 30), (259200, 1, 8640, 288), torch.int8)
        # Topologically Sorted Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_33.run(buf104, buf111, buf112, 3600, 288, grid=grid(3600, 288), stream=stream0)
        buf113 = reinterpret_tensor(buf115, (4, 384, 30, 30), (691200, 900, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_54, branch3x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf105, primals_133, primals_134, primals_135, primals_136, buf113, 1536, 900, grid=grid(1536, 900), stream=stream0)
        buf114 = reinterpret_tensor(buf115, (4, 96, 30, 30), (691200, 900, 30, 1), 345600)  # alias
        # Topologically Sorted Source Nodes: [input_60, branch3x3dbl_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf110, primals_148, primals_149, primals_150, primals_151, buf114, 384, 900, grid=grid(384, 900), stream=stream0)
        buf116 = empty_strided_cuda((4, 768, 30, 30), (691200, 1, 23040, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf115, buf116, 3072, 900, grid=grid(3072, 900), stream=stream0)
        del buf111
        del buf113
        del buf114
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf116, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf119 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_64, branch7x7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf118, primals_158, primals_159, primals_160, primals_161, buf119, 460800, grid=grid(460800), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, buf17, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf121 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_66, branch7x7_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf120, primals_163, primals_164, primals_165, primals_166, buf121, 460800, grid=grid(460800), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, buf18, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf116, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf124 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_70, branch7x7dbl], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf123, primals_173, primals_174, primals_175, primals_176, buf124, 460800, grid=grid(460800), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, buf19, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf126 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_72, branch7x7dbl_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf125, primals_178, primals_179, primals_180, primals_181, buf126, 460800, grid=grid(460800), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, buf20, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf128 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_74, branch7x7dbl_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf127, primals_183, primals_184, primals_185, primals_186, buf128, 460800, grid=grid(460800), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, buf21, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 128, 30, 30), (115200, 1, 3840, 128))
        buf130 = empty_strided_cuda((4, 128, 30, 30), (115200, 1, 3840, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_76, branch7x7dbl_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf129, primals_188, primals_189, primals_190, primals_191, buf130, 460800, grid=grid(460800), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, buf22, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf132 = reinterpret_tensor(buf115, (4, 768, 30, 30), (691200, 1, 23040, 768), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [branch_pool_7], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf116, buf132, 2764800, grid=grid(2764800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf134 = empty_strided_cuda((4, 768, 30, 30), (691200, 1, 23040, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_39.run(buf117, primals_153, primals_154, primals_155, primals_156, buf122, primals_168, primals_169, primals_170, primals_171, buf131, primals_193, primals_194, primals_195, primals_196, buf133, primals_198, primals_199, primals_200, primals_201, buf134, 2764800, grid=grid(2764800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf134, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf137 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_84, branch7x7_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf136, primals_208, primals_209, primals_210, primals_211, buf137, 576000, grid=grid(576000), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, buf23, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf139 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_86, branch7x7_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf138, primals_213, primals_214, primals_215, primals_216, buf139, 576000, grid=grid(576000), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, buf24, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf134, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf142 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_90, branch7x7dbl_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf141, primals_223, primals_224, primals_225, primals_226, buf142, 576000, grid=grid(576000), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, buf25, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf144 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_92, branch7x7dbl_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf143, primals_228, primals_229, primals_230, primals_231, buf144, 576000, grid=grid(576000), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, buf26, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf146 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_94, branch7x7dbl_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf145, primals_233, primals_234, primals_235, primals_236, buf146, 576000, grid=grid(576000), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, buf27, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf148 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_96, branch7x7dbl_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf147, primals_238, primals_239, primals_240, primals_241, buf148, 576000, grid=grid(576000), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [input_97], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, buf28, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf150 = empty_strided_cuda((4, 768, 30, 30), (691200, 1, 23040, 768), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_9], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf134, buf150, 2764800, grid=grid(2764800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf152 = empty_strided_cuda((4, 768, 30, 30), (691200, 1, 23040, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_39.run(buf135, primals_203, primals_204, primals_205, primals_206, buf140, primals_218, primals_219, primals_220, primals_221, buf149, primals_243, primals_244, primals_245, primals_246, buf151, primals_248, primals_249, primals_250, primals_251, buf152, 2764800, grid=grid(2764800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_101], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf152, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf155 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_104, branch7x7_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf154, primals_258, primals_259, primals_260, primals_261, buf155, 576000, grid=grid(576000), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [input_105], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, buf29, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf157 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_106, branch7x7_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf156, primals_263, primals_264, primals_265, primals_266, buf157, 576000, grid=grid(576000), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [input_107], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, buf30, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [input_109], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf152, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf160 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_110, branch7x7dbl_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf159, primals_273, primals_274, primals_275, primals_276, buf160, 576000, grid=grid(576000), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [input_111], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, buf31, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf162 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_112, branch7x7dbl_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf161, primals_278, primals_279, primals_280, primals_281, buf162, 576000, grid=grid(576000), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [input_113], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, buf32, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf164 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_114, branch7x7dbl_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf163, primals_283, primals_284, primals_285, primals_286, buf164, 576000, grid=grid(576000), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, buf33, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 160, 30, 30), (144000, 1, 4800, 160))
        buf166 = empty_strided_cuda((4, 160, 30, 30), (144000, 1, 4800, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_116, branch7x7dbl_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf165, primals_288, primals_289, primals_290, primals_291, buf166, 576000, grid=grid(576000), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, buf34, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf168 = empty_strided_cuda((4, 768, 30, 30), (691200, 1, 23040, 768), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_11], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf152, buf168, 2764800, grid=grid(2764800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf170 = empty_strided_cuda((4, 768, 30, 30), (691200, 1, 23040, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_39.run(buf153, primals_253, primals_254, primals_255, primals_256, buf158, primals_268, primals_269, primals_270, primals_271, buf167, primals_293, primals_294, primals_295, primals_296, buf169, primals_298, primals_299, primals_300, primals_301, buf170, 2764800, grid=grid(2764800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_121], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [input_123], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf170, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf173 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_124, branch7x7_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf172, primals_308, primals_309, primals_310, primals_311, buf173, 691200, grid=grid(691200), stream=stream0)
        del primals_311
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, buf35, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf175 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_126, branch7x7_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf174, primals_313, primals_314, primals_315, primals_316, buf175, 691200, grid=grid(691200), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [input_127], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, buf36, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 192, 30, 30), (172800, 1, 5760, 192))
        # Topologically Sorted Source Nodes: [input_129], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf170, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf178 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_130, branch7x7dbl_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf177, primals_323, primals_324, primals_325, primals_326, buf178, 691200, grid=grid(691200), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, buf37, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf180 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_132, branch7x7dbl_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf179, primals_328, primals_329, primals_330, primals_331, buf180, 691200, grid=grid(691200), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [input_133], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, buf38, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf182 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_134, branch7x7dbl_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf181, primals_333, primals_334, primals_335, primals_336, buf182, 691200, grid=grid(691200), stream=stream0)
        del primals_336
        # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, buf39, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf184 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_136, branch7x7dbl_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf183, primals_338, primals_339, primals_340, primals_341, buf184, 691200, grid=grid(691200), stream=stream0)
        del primals_341
        # Topologically Sorted Source Nodes: [input_137], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, buf40, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf186 = empty_strided_cuda((4, 768, 30, 30), (691200, 1, 23040, 768), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_13], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf170, buf186, 2764800, grid=grid(2764800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf188 = empty_strided_cuda((4, 768, 30, 30), (691200, 1, 23040, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_39.run(buf171, primals_303, primals_304, primals_305, primals_306, buf176, primals_318, primals_319, primals_320, primals_321, buf185, primals_343, primals_344, primals_345, primals_346, buf187, primals_348, primals_349, primals_350, primals_351, buf188, 2764800, grid=grid(2764800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_141], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_352, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf190 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_142, branch3x3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf189, primals_353, primals_354, primals_355, primals_356, buf190, 691200, grid=grid(691200), stream=stream0)
        del primals_356
        # Topologically Sorted Source Nodes: [input_143], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, buf41, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 320, 14, 14), (62720, 1, 4480, 320))
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf188, primals_362, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf193 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_146, branch7x7x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf192, primals_363, primals_364, primals_365, primals_366, buf193, 691200, grid=grid(691200), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, buf42, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf195 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_148, branch7x7x3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf194, primals_368, primals_369, primals_370, primals_371, buf195, 691200, grid=grid(691200), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, buf43, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 192, 30, 30), (172800, 1, 5760, 192))
        buf197 = empty_strided_cuda((4, 192, 30, 30), (172800, 1, 5760, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_150, branch7x7x3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf196, primals_373, primals_374, primals_375, primals_376, buf197, 691200, grid=grid(691200), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [input_151], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, buf44, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf203 = empty_strided_cuda((4, 1280, 14, 14), (250880, 196, 14, 1), torch.float32)
        buf199 = reinterpret_tensor(buf203, (4, 768, 14, 14), (250880, 196, 14, 1), 100352)  # alias
        buf200 = empty_strided_cuda((4, 768, 14, 14), (150528, 1, 10752, 768), torch.int8)
        # Topologically Sorted Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_42.run(buf188, buf199, buf200, 784, 768, grid=grid(784, 768), stream=stream0)
        buf201 = reinterpret_tensor(buf203, (4, 320, 14, 14), (250880, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_144, branch3x3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf191, primals_358, primals_359, primals_360, primals_361, buf201, 1280, 196, grid=grid(1280, 196), stream=stream0)
        buf202 = reinterpret_tensor(buf203, (4, 192, 14, 14), (250880, 196, 14, 1), 62720)  # alias
        # Topologically Sorted Source Nodes: [input_152, branch7x7x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf198, primals_378, primals_379, primals_380, primals_381, buf202, 768, 196, grid=grid(768, 196), stream=stream0)
        buf204 = empty_strided_cuda((4, 1280, 14, 14), (250880, 1, 17920, 1280), torch.float32)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_45.run(buf203, buf204, 5120, 196, grid=grid(5120, 196), stream=stream0)
        del buf199
        del buf201
        del buf202
        # Topologically Sorted Source Nodes: [input_153], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_382, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 320, 14, 14), (62720, 1, 4480, 320))
        # Topologically Sorted Source Nodes: [input_155], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf204, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 384, 14, 14), (75264, 1, 5376, 384))
        buf207 = empty_strided_cuda((4, 384, 14, 14), (75264, 1, 5376, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_156, branch3x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf206, primals_388, primals_389, primals_390, primals_391, buf207, 301056, grid=grid(301056), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [input_157], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, buf45, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 384, 14, 14), (75264, 1, 5376, 384))
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf207, buf46, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 384, 14, 14), (75264, 1, 5376, 384))
        buf222 = empty_strided_cuda((4, 2048, 14, 14), (401408, 196, 14, 1), torch.float32)
        buf210 = reinterpret_tensor(buf222, (4, 768, 14, 14), (401408, 196, 14, 1), 62720)  # alias
        # Topologically Sorted Source Nodes: [branch3x3_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_47.run(buf208, primals_393, primals_394, primals_395, primals_396, buf209, primals_398, primals_399, primals_400, primals_401, buf210, 602112, grid=grid(602112), stream=stream0)
        # Topologically Sorted Source Nodes: [input_161], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf204, primals_402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 448, 14, 14), (87808, 1, 6272, 448))
        buf212 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [input_162, branch3x3dbl_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf211, primals_403, primals_404, primals_405, primals_406, buf212, 351232, grid=grid(351232), stream=stream0)
        del primals_406
        # Topologically Sorted Source Nodes: [input_163], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 384, 14, 14), (75264, 1, 5376, 384))
        buf214 = empty_strided_cuda((4, 384, 14, 14), (75264, 1, 5376, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_164, branch3x3dbl_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf213, primals_408, primals_409, primals_410, primals_411, buf214, 301056, grid=grid(301056), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [input_165], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, buf48, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 384, 14, 14), (75264, 1, 5376, 384))
        # Topologically Sorted Source Nodes: [input_167], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf214, buf49, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 384, 14, 14), (75264, 1, 5376, 384))
        buf217 = reinterpret_tensor(buf222, (4, 768, 14, 14), (401408, 196, 14, 1), 213248)  # alias
        # Topologically Sorted Source Nodes: [branch3x3dbl_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_47.run(buf215, primals_413, primals_414, primals_415, primals_416, buf216, primals_418, primals_419, primals_420, primals_421, buf217, 602112, grid=grid(602112), stream=stream0)
        buf218 = reinterpret_tensor(buf203, (4, 1280, 14, 14), (250880, 1, 17920, 1280), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [branch_pool_16], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_49.run(buf204, buf218, 1003520, grid=grid(1003520), stream=stream0)
        # Topologically Sorted Source Nodes: [input_169], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf220 = reinterpret_tensor(buf222, (4, 320, 14, 14), (401408, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_154, branch1x1_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf205, primals_383, primals_384, primals_385, primals_386, buf220, 1280, 196, grid=grid(1280, 196), stream=stream0)
        buf221 = reinterpret_tensor(buf222, (4, 192, 14, 14), (401408, 196, 14, 1), 363776)  # alias
        # Topologically Sorted Source Nodes: [input_170, branch_pool_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf219, primals_423, primals_424, primals_425, primals_426, buf221, 768, 196, grid=grid(768, 196), stream=stream0)
        buf223 = empty_strided_cuda((4, 2048, 14, 14), (401408, 1, 28672, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_52.run(buf222, buf223, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf210
        del buf217
        del buf220
        del buf221
        # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_427, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 320, 14, 14), (62720, 1, 4480, 320))
        # Topologically Sorted Source Nodes: [input_173], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf223, primals_432, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 384, 14, 14), (75264, 1, 5376, 384))
        buf226 = empty_strided_cuda((4, 384, 14, 14), (75264, 1, 5376, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_174, branch3x3_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf225, primals_433, primals_434, primals_435, primals_436, buf226, 301056, grid=grid(301056), stream=stream0)
        del primals_436
        # Topologically Sorted Source Nodes: [input_175], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, buf50, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 384, 14, 14), (75264, 1, 5376, 384))
        # Topologically Sorted Source Nodes: [input_177], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf51, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 384, 14, 14), (75264, 1, 5376, 384))
        buf241 = buf222; del buf222  # reuse
        buf229 = reinterpret_tensor(buf241, (4, 768, 14, 14), (401408, 196, 14, 1), 62720)  # alias
        # Topologically Sorted Source Nodes: [branch3x3_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_47.run(buf227, primals_438, primals_439, primals_440, primals_441, buf228, primals_443, primals_444, primals_445, primals_446, buf229, 602112, grid=grid(602112), stream=stream0)
        # Topologically Sorted Source Nodes: [input_179], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf223, primals_447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 448, 14, 14), (87808, 1, 6272, 448))
        buf231 = empty_strided_cuda((4, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [input_180, branch3x3dbl_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf230, primals_448, primals_449, primals_450, primals_451, buf231, 351232, grid=grid(351232), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [input_181], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 384, 14, 14), (75264, 1, 5376, 384))
        buf233 = empty_strided_cuda((4, 384, 14, 14), (75264, 1, 5376, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_182, branch3x3dbl_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf232, primals_453, primals_454, primals_455, primals_456, buf233, 301056, grid=grid(301056), stream=stream0)
        del primals_456
        # Topologically Sorted Source Nodes: [input_183], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, buf53, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 384, 14, 14), (75264, 1, 5376, 384))
        # Topologically Sorted Source Nodes: [input_185], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf233, buf54, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 384, 14, 14), (75264, 1, 5376, 384))
        buf236 = reinterpret_tensor(buf241, (4, 768, 14, 14), (401408, 196, 14, 1), 213248)  # alias
        # Topologically Sorted Source Nodes: [branch3x3dbl_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_47.run(buf234, primals_458, primals_459, primals_460, primals_461, buf235, primals_463, primals_464, primals_465, primals_466, buf236, 602112, grid=grid(602112), stream=stream0)
        buf237 = empty_strided_cuda((4, 2048, 14, 14), (401408, 1, 28672, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_18], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_53.run(buf223, buf237, 1605632, grid=grid(1605632), stream=stream0)
        # Topologically Sorted Source Nodes: [input_187], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_467, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 192, 14, 14), (37632, 1, 2688, 192))
        buf239 = reinterpret_tensor(buf241, (4, 320, 14, 14), (401408, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_172, branch1x1_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf224, primals_428, primals_429, primals_430, primals_431, buf239, 1280, 196, grid=grid(1280, 196), stream=stream0)
        buf240 = reinterpret_tensor(buf241, (4, 192, 14, 14), (401408, 196, 14, 1), 363776)  # alias
        # Topologically Sorted Source Nodes: [input_188, branch_pool_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf238, primals_468, primals_469, primals_470, primals_471, buf240, 768, 196, grid=grid(768, 196), stream=stream0)
        buf242 = empty_strided_cuda((4, 2048, 14, 14), (401408, 1, 28672, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_52.run(buf241, buf242, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf229
        del buf236
        del buf239
        del buf240
        del buf241
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.avg_pool2d]
        buf243 = torch.ops.aten.avg_pool2d.default(buf242, [8, 8], [8, 8], [0, 0], False, True, None)
        buf244 = buf243
        del buf243
        buf245 = empty_strided_cuda((4, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_473, reinterpret_tensor(buf244, (4, 2048), (2048, 1), 0), reinterpret_tensor(primals_472, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf245)
        del primals_473
    return (buf245, buf0, buf1, primals_3, primals_4, primals_5, buf2, primals_8, primals_9, primals_10, buf3, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, buf4, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, buf5, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, buf6, primals_48, primals_49, primals_50, buf7, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, buf8, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, buf9, primals_83, primals_84, primals_85, buf10, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, buf11, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, buf12, primals_118, primals_119, primals_120, buf13, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, buf14, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, buf15, primals_143, primals_144, primals_145, buf16, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, buf17, primals_163, primals_164, primals_165, buf18, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, buf19, primals_178, primals_179, primals_180, buf20, primals_183, primals_184, primals_185, buf21, primals_188, primals_189, primals_190, buf22, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, buf23, primals_213, primals_214, primals_215, buf24, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, buf25, primals_228, primals_229, primals_230, buf26, primals_233, primals_234, primals_235, buf27, primals_238, primals_239, primals_240, buf28, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, buf29, primals_263, primals_264, primals_265, buf30, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, buf31, primals_278, primals_279, primals_280, buf32, primals_283, primals_284, primals_285, buf33, primals_288, primals_289, primals_290, buf34, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, buf35, primals_313, primals_314, primals_315, buf36, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, buf37, primals_328, primals_329, primals_330, buf38, primals_333, primals_334, primals_335, buf39, primals_338, primals_339, primals_340, buf40, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, buf41, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, buf42, primals_368, primals_369, primals_370, buf43, primals_373, primals_374, primals_375, buf44, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, buf45, primals_393, primals_394, primals_395, primals_396, buf46, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, buf47, primals_408, primals_409, primals_410, buf48, primals_413, primals_414, primals_415, primals_416, buf49, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, buf50, primals_438, primals_439, primals_440, primals_441, buf51, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, buf52, primals_453, primals_454, primals_455, buf53, primals_458, primals_459, primals_460, primals_461, buf54, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf112, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf128, buf129, buf130, buf131, buf132, buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf193, buf194, buf195, buf196, buf197, buf198, buf200, buf204, buf205, buf206, buf207, buf208, buf209, buf211, buf212, buf213, buf214, buf215, buf216, buf218, buf219, buf223, buf224, buf225, buf226, buf227, buf228, buf230, buf231, buf232, buf233, buf234, buf235, buf237, buf238, buf242, reinterpret_tensor(buf244, (4, 2048), (2048, 1), 0), primals_472, )


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
    primals_472 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
