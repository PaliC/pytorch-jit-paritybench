# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/xe/cxetavjgioi3tnju2b2qja3zfdzksswoyprex3wmsbtiahic4uln.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 12)
    y1 = yindex // 12
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 12*x2 + 108*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m7/cm7g5ayrrca47y3n5bll6ozmn62mnjjxjd77fef7j7lvniua7qzu.py
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
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/tz/ctzbtkvkyehfxzarvl2p7gdtsmsvqiguaxydkrybi3ra7uykppmp.py
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/gj/cgjooydkdcebijda33i4wcnp6j3jmarypxur25bgj6fhivapr2kb.py
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
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/5p/c5p7fkld2iviljxa6a65wn4rmxoomtzu6jjj4itwoev4vwd6uvdh.py
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
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/jn/cjn4obaqq242lac34mkqqfxd3vig4q3amnq2tfmbawyeisy4cl6u.py
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
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x2 + 9216*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/za/czan3t62dyzozls4co6hn756pu3xhyjiqr4vetc5i3pkl7hmka3x.py
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
    size_hints={'y': 1048576, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x2 + 9216*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xj/cxjinqx4r5kkhqielr5jyccg5eaay67tit6mglf364bpyxlip4ab.py
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
    size_hints={'y': 8388608, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8388608
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 2048*x2 + 18432*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/fe/cfed47gmdfmhvpkyo6eakqmpvdwm62cg3gkpc4fa6cfqfuckbjuk.py
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
    size_hints={'y': 4194304, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4194304
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 2048*x2 + 18432*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/2y/c2y4taieuipv24fli4kmhjyao4sbrfhshs6ezpqv7btjjpahlthl.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_2, %slice_6, %slice_4, %slice_8], 1), kwargs = {})
triton_poi_fused_cat_9 = async_compile.triton('triton_poi_fused_cat_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x1 = ((xindex // 12) % 2)
    x2 = ((xindex // 24) % 2)
    x3 = xindex // 48
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2*x1 + 8*x2 + 16*(x0) + 48*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 6, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (4 + 2*x1 + 8*x2 + 16*((-3) + x0) + 48*x3), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 9, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr0 + (1 + 2*x1 + 8*x2 + 16*((-6) + x0) + 48*x3), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 12, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr0 + (5 + 2*x1 + 8*x2 + 16*((-9) + x0) + 48*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x4), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uk/cukj3mlywzwgtv6olp6b4c7k4wp3p5vr5vlb3tfnjhduhbj7kdib.py
# Topologically Sorted Source Nodes: [batch_norm, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   x_1 => mul_3, sigmoid
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/qo/cqorhjzw4cxxms3hmwqdz75bo7ujlm7tvdabbtddsbuxwlzan27f.py
# Topologically Sorted Source Nodes: [batch_norm_1, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_5, mul_6, sub_1
#   input_1 => mul_7, sigmoid_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_7 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bq/cbqan6yfos6lmmpe2qkgb7g7o2jc3rxuzknay7kqxdjl6sbvyrj2.py
# Topologically Sorted Source Nodes: [batch_norm_2, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_2 => add_5, mul_10, mul_9, sub_2
#   x_2 => mul_11, sigmoid_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_5,), kwargs = {})
#   %mul_11 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %sigmoid_2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m6/cm6ixqprsen3adqshrfcf4vpowurmxzfrdfulb3xirbul4wj5vyp.py
# Topologically Sorted Source Nodes: [batch_norm_3], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_3 => add_7, mul_13, mul_14, sub_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vl/cvlxfunfsijxswvlxtshiqtc3oetr2jgue5y4yvjf57e6eqrd76l.py
# Topologically Sorted Source Nodes: [batch_norm_5, y, y_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_5 => add_11, mul_21, mul_22, sub_5
#   y => mul_23, sigmoid_5
#   y_1 => add_12
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_47), kwargs = {})
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_5), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %mul_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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
    tmp18 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tu/ctu7ojvlbjpumw6mgjp4krbyyswxrjweogfcr3o2rocaroarmy3m.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_4 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_67, %mul_15], 1), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_poi_fused_cat_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr1 + (256*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 512, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (256*x1 + ((-256) + x0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp11, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7g/c7gsb6r5ggs2waic5w7jo2fwtqfhq5w74orlbuspjb5bfzbef4ld.py
# Topologically Sorted Source Nodes: [batch_norm_29, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_29 => add_71, mul_117, mul_118, sub_29
#   input_3 => mul_119, sigmoid_29
# Graph fragment:
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_117, %unsqueeze_237), kwargs = {})
#   %add_71 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_118, %unsqueeze_239), kwargs = {})
#   %sigmoid_29 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_71,), kwargs = {})
#   %mul_119 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_71, %sigmoid_29), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/gn/cgnwnssya6wb6kdtdxbq6ahyjxzdhogfjn3dijxud2ko4fxfrisy.py
# Topologically Sorted Source Nodes: [batch_norm_31], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_31 => add_75, mul_125, mul_126, sub_31
# Graph fragment:
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_249), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_125, %unsqueeze_253), kwargs = {})
#   %add_75 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_126, %unsqueeze_255), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rh/crh5auwc3jbewqdu4742d3y3alz6tlyxdfsljjm6yx67biaodolu.py
# Topologically Sorted Source Nodes: [batch_norm_33, y_24, y_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_33 => add_79, mul_133, mul_134, sub_33
#   y_24 => mul_135, sigmoid_33
#   y_25 => add_80
# Graph fragment:
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %unsqueeze_269), kwargs = {})
#   %add_79 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %unsqueeze_271), kwargs = {})
#   %sigmoid_33 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sigmoid_33), kwargs = {})
#   %add_80 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_135, %mul_123), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7l/c7llbchdnpz7uvq7bzdals2fyhe4rzq5gf4bbrwmncnmyjjvtgsh.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_7 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_255, %mul_127], 1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
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
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr1 + (512*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1024, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (512*x1 + ((-512) + x0)), tmp12, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp11, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/pf/cpfft5ljqutve77dnor2uzeuc7oqydlqyxo4g3wwgr4ruim6cygc.py
# Topologically Sorted Source Nodes: [batch_norm_105, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_105 => add_259, mul_421, mul_422, sub_105
#   input_5 => mul_423, sigmoid_105
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_841), kwargs = {})
#   %mul_421 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_422 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_421, %unsqueeze_845), kwargs = {})
#   %add_259 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_422, %unsqueeze_847), kwargs = {})
#   %sigmoid_105 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_259,), kwargs = {})
#   %mul_423 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_259, %sigmoid_105), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/xm/cxmueinejz6yczsvuwygzo3mp6rjaeizmk4xqd5jxp7fxyxcash3.py
# Topologically Sorted Source Nodes: [batch_norm_107], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_107 => add_263, mul_429, mul_430, sub_107
# Graph fragment:
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_857), kwargs = {})
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_429, %unsqueeze_861), kwargs = {})
#   %add_263 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_430, %unsqueeze_863), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/3f/c3fmaudshurqp4k77aes53eh4bt7glhjkcbzqbxxbuvcrbkumqph.py
# Topologically Sorted Source Nodes: [batch_norm_109, y_96, y_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_109 => add_267, mul_437, mul_438, sub_109
#   y_96 => mul_439, sigmoid_109
#   y_97 => add_268
# Graph fragment:
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_109, %unsqueeze_873), kwargs = {})
#   %mul_437 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %unsqueeze_875), kwargs = {})
#   %mul_438 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_437, %unsqueeze_877), kwargs = {})
#   %add_267 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_438, %unsqueeze_879), kwargs = {})
#   %sigmoid_109 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_267,), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_267, %sigmoid_109), kwargs = {})
#   %add_268 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_439, %mul_427), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
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
    tmp18 = tl.load(in_ptr5 + (x2), None)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/25/c25pa2ddamrpohf6zrxo5afh7e4hmrirptyx5flpleqjqlniczfw.py
# Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_10 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_443, %mul_431], 1), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1024*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr1 + (1024*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 2048, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (1024*x1 + ((-1024) + x0)), tmp12, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp11, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/dn/cdnj4zhqmwlg6v64j2fiwdtmsbjrlxkmxxf7ijel3e2dnj5yixhl.py
# Topologically Sorted Source Nodes: [batch_norm_181, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_181 => add_447, mul_725, mul_726, sub_181
#   input_7 => mul_727, sigmoid_181
# Graph fragment:
#   %sub_181 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_181, %unsqueeze_1449), kwargs = {})
#   %mul_725 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_181, %unsqueeze_1451), kwargs = {})
#   %mul_726 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_725, %unsqueeze_1453), kwargs = {})
#   %add_447 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_726, %unsqueeze_1455), kwargs = {})
#   %sigmoid_181 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_447,), kwargs = {})
#   %mul_727 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_447, %sigmoid_181), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4096)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/xj/cxj5jtljwxfi7m6klsl4jveyw2aeoxpisbf6kuvjuoigxn7qucgj.py
# Topologically Sorted Source Nodes: [batch_norm_182, x_11, max_pool2d], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   batch_norm_182 => add_449, mul_729, mul_730, sub_182
#   max_pool2d => getitem_1
#   x_11 => mul_731, sigmoid_182
# Graph fragment:
#   %sub_182 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_182, %unsqueeze_1457), kwargs = {})
#   %mul_729 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_182, %unsqueeze_1459), kwargs = {})
#   %mul_730 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_729, %unsqueeze_1461), kwargs = {})
#   %add_449 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_730, %unsqueeze_1463), kwargs = {})
#   %sigmoid_182 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_449,), kwargs = {})
#   %mul_731 : [num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_449, %sigmoid_182), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_silu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_silu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_silu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_silu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full([1], -2, tl.int64)
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp18 >= tmp19
    tmp21 = tl.full([1], 1, tl.int64)
    tmp22 = tmp18 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tmp23 & tmp23
    tmp25 = tl.full([1], -1, tl.int64)
    tmp26 = tmp25 >= tmp19
    tmp27 = tmp25 < tmp21
    tmp28 = tmp26 & tmp27
    tmp29 = tmp23 & tmp28
    tmp30 = tmp17 > tmp17
    tmp31 = tl.full([1], 1, tl.int8)
    tmp32 = tl.full([1], 0, tl.int8)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = triton_helpers.maximum(tmp17, tmp17)
    tmp35 = tmp19 >= tmp19
    tmp36 = tmp19 < tmp21
    tmp37 = tmp35 & tmp36
    tmp38 = tmp23 & tmp37
    tmp39 = tmp17 > tmp34
    tmp40 = tl.full([1], 2, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp33)
    tmp42 = triton_helpers.maximum(tmp17, tmp34)
    tmp43 = tmp21 >= tmp19
    tmp44 = tmp21 < tmp21
    tmp45 = tmp43 & tmp44
    tmp46 = tmp23 & tmp45
    tmp47 = tmp17 > tmp42
    tmp48 = tl.full([1], 3, tl.int8)
    tmp49 = tl.where(tmp47, tmp48, tmp41)
    tmp50 = triton_helpers.maximum(tmp17, tmp42)
    tmp51 = tl.full([1], 2, tl.int64)
    tmp52 = tmp51 >= tmp19
    tmp53 = tmp51 < tmp21
    tmp54 = tmp52 & tmp53
    tmp55 = tmp23 & tmp54
    tmp56 = tmp17 > tmp50
    tmp57 = tl.full([1], 4, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp49)
    tmp59 = triton_helpers.maximum(tmp17, tmp50)
    tmp60 = tmp28 & tmp23
    tmp61 = tmp17 > tmp59
    tmp62 = tl.full([1], 5, tl.int8)
    tmp63 = tl.where(tmp61, tmp62, tmp58)
    tmp64 = triton_helpers.maximum(tmp17, tmp59)
    tmp65 = tmp28 & tmp28
    tmp66 = tmp17 > tmp64
    tmp67 = tl.full([1], 6, tl.int8)
    tmp68 = tl.where(tmp66, tmp67, tmp63)
    tmp69 = triton_helpers.maximum(tmp17, tmp64)
    tmp70 = tmp28 & tmp37
    tmp71 = tmp17 > tmp69
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp68)
    tmp74 = triton_helpers.maximum(tmp17, tmp69)
    tmp75 = tmp28 & tmp45
    tmp76 = tmp17 > tmp74
    tmp77 = tl.full([1], 8, tl.int8)
    tmp78 = tl.where(tmp76, tmp77, tmp73)
    tmp79 = triton_helpers.maximum(tmp17, tmp74)
    tmp80 = tmp28 & tmp54
    tmp81 = tmp17 > tmp79
    tmp82 = tl.full([1], 9, tl.int8)
    tmp83 = tl.where(tmp81, tmp82, tmp78)
    tmp84 = triton_helpers.maximum(tmp17, tmp79)
    tmp85 = tmp37 & tmp23
    tmp86 = tmp17 > tmp84
    tmp87 = tl.full([1], 10, tl.int8)
    tmp88 = tl.where(tmp86, tmp87, tmp83)
    tmp89 = triton_helpers.maximum(tmp17, tmp84)
    tmp90 = tmp37 & tmp28
    tmp91 = tmp17 > tmp89
    tmp92 = tl.full([1], 11, tl.int8)
    tmp93 = tl.where(tmp91, tmp92, tmp88)
    tmp94 = triton_helpers.maximum(tmp17, tmp89)
    tmp95 = tmp37 & tmp37
    tmp96 = tmp17 > tmp94
    tmp97 = tl.full([1], 12, tl.int8)
    tmp98 = tl.where(tmp96, tmp97, tmp93)
    tmp99 = triton_helpers.maximum(tmp17, tmp94)
    tmp100 = tmp37 & tmp45
    tmp101 = tmp17 > tmp99
    tmp102 = tl.full([1], 13, tl.int8)
    tmp103 = tl.where(tmp101, tmp102, tmp98)
    tmp104 = triton_helpers.maximum(tmp17, tmp99)
    tmp105 = tmp37 & tmp54
    tmp106 = tmp17 > tmp104
    tmp107 = tl.full([1], 14, tl.int8)
    tmp108 = tl.where(tmp106, tmp107, tmp103)
    tmp109 = triton_helpers.maximum(tmp17, tmp104)
    tmp110 = tmp45 & tmp23
    tmp111 = tmp17 > tmp109
    tmp112 = tl.full([1], 15, tl.int8)
    tmp113 = tl.where(tmp111, tmp112, tmp108)
    tmp114 = triton_helpers.maximum(tmp17, tmp109)
    tmp115 = tmp45 & tmp28
    tmp116 = tmp17 > tmp114
    tmp117 = tl.full([1], 16, tl.int8)
    tmp118 = tl.where(tmp116, tmp117, tmp113)
    tmp119 = triton_helpers.maximum(tmp17, tmp114)
    tmp120 = tmp45 & tmp37
    tmp121 = tmp17 > tmp119
    tmp122 = tl.full([1], 17, tl.int8)
    tmp123 = tl.where(tmp121, tmp122, tmp118)
    tmp124 = triton_helpers.maximum(tmp17, tmp119)
    tmp125 = tmp45 & tmp45
    tmp126 = tmp17 > tmp124
    tmp127 = tl.full([1], 18, tl.int8)
    tmp128 = tl.where(tmp126, tmp127, tmp123)
    tmp129 = triton_helpers.maximum(tmp17, tmp124)
    tmp130 = tmp45 & tmp54
    tmp131 = tmp17 > tmp129
    tmp132 = tl.full([1], 19, tl.int8)
    tmp133 = tl.where(tmp131, tmp132, tmp128)
    tmp134 = triton_helpers.maximum(tmp17, tmp129)
    tmp135 = tmp54 & tmp23
    tmp136 = tmp17 > tmp134
    tmp137 = tl.full([1], 20, tl.int8)
    tmp138 = tl.where(tmp136, tmp137, tmp133)
    tmp139 = triton_helpers.maximum(tmp17, tmp134)
    tmp140 = tmp54 & tmp28
    tmp141 = tmp17 > tmp139
    tmp142 = tl.full([1], 21, tl.int8)
    tmp143 = tl.where(tmp141, tmp142, tmp138)
    tmp144 = triton_helpers.maximum(tmp17, tmp139)
    tmp145 = tmp54 & tmp37
    tmp146 = tmp17 > tmp144
    tmp147 = tl.full([1], 22, tl.int8)
    tmp148 = tl.where(tmp146, tmp147, tmp143)
    tmp149 = triton_helpers.maximum(tmp17, tmp144)
    tmp150 = tmp54 & tmp45
    tmp151 = tmp17 > tmp149
    tmp152 = tl.full([1], 23, tl.int8)
    tmp153 = tl.where(tmp151, tmp152, tmp148)
    tmp154 = triton_helpers.maximum(tmp17, tmp149)
    tmp155 = tmp54 & tmp54
    tmp156 = tmp17 > tmp154
    tmp157 = tl.full([1], 24, tl.int8)
    tmp158 = tl.where(tmp156, tmp157, tmp153)
    tmp159 = triton_helpers.maximum(tmp17, tmp154)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr0 + (x2), tmp158, None)
''', device_str='cuda')


# kernel path: inductor_cache/co/ccoey2vw554ojpicylxccyt4bko4jqjfndzbqcfutet5dawsor5g.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_12 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_731, %getitem, %getitem_2, %getitem_4], 1), kwargs = {})
triton_poi_fused_cat_26 = async_compile.triton('triton_poi_fused_cat_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 28, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8192)
    x1 = xindex // 8192
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 4096, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.full([1], -2, tl.int64)
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 1, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp15
    tmp17 = tmp16 & tmp9
    tmp18 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp17, eviction_policy='evict_last', other=float("-inf"))
    tmp19 = tl.full([1], -1, tl.int64)
    tmp20 = tmp19 >= tmp11
    tmp21 = tmp19 < tmp13
    tmp22 = tmp20 & tmp21
    tmp23 = tmp15 & tmp22
    tmp24 = tmp23 & tmp9
    tmp25 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp24, eviction_policy='evict_last', other=float("-inf"))
    tmp26 = triton_helpers.maximum(tmp25, tmp18)
    tmp27 = tmp11 >= tmp11
    tmp28 = tmp11 < tmp13
    tmp29 = tmp27 & tmp28
    tmp30 = tmp15 & tmp29
    tmp31 = tmp30 & tmp9
    tmp32 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp31, eviction_policy='evict_last', other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp26)
    tmp34 = tmp13 >= tmp11
    tmp35 = tmp13 < tmp13
    tmp36 = tmp34 & tmp35
    tmp37 = tmp15 & tmp36
    tmp38 = tmp37 & tmp9
    tmp39 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp38, eviction_policy='evict_last', other=float("-inf"))
    tmp40 = triton_helpers.maximum(tmp39, tmp33)
    tmp41 = tl.full([1], 2, tl.int64)
    tmp42 = tmp41 >= tmp11
    tmp43 = tmp41 < tmp13
    tmp44 = tmp42 & tmp43
    tmp45 = tmp15 & tmp44
    tmp46 = tmp45 & tmp9
    tmp47 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp40)
    tmp49 = tmp22 & tmp15
    tmp50 = tmp49 & tmp9
    tmp51 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp50, eviction_policy='evict_last', other=float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp48)
    tmp53 = tmp22 & tmp22
    tmp54 = tmp53 & tmp9
    tmp55 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp54, eviction_policy='evict_last', other=float("-inf"))
    tmp56 = triton_helpers.maximum(tmp55, tmp52)
    tmp57 = tmp22 & tmp29
    tmp58 = tmp57 & tmp9
    tmp59 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp58, eviction_policy='evict_last', other=float("-inf"))
    tmp60 = triton_helpers.maximum(tmp59, tmp56)
    tmp61 = tmp22 & tmp36
    tmp62 = tmp61 & tmp9
    tmp63 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp62, eviction_policy='evict_last', other=float("-inf"))
    tmp64 = triton_helpers.maximum(tmp63, tmp60)
    tmp65 = tmp22 & tmp44
    tmp66 = tmp65 & tmp9
    tmp67 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp66, eviction_policy='evict_last', other=float("-inf"))
    tmp68 = triton_helpers.maximum(tmp67, tmp64)
    tmp69 = tmp29 & tmp15
    tmp70 = tmp69 & tmp9
    tmp71 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp70, eviction_policy='evict_last', other=float("-inf"))
    tmp72 = triton_helpers.maximum(tmp71, tmp68)
    tmp73 = tmp29 & tmp22
    tmp74 = tmp73 & tmp9
    tmp75 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp74, eviction_policy='evict_last', other=float("-inf"))
    tmp76 = triton_helpers.maximum(tmp75, tmp72)
    tmp77 = tmp29 & tmp29
    tmp78 = tmp77 & tmp9
    tmp79 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp78, eviction_policy='evict_last', other=float("-inf"))
    tmp80 = triton_helpers.maximum(tmp79, tmp76)
    tmp81 = tmp29 & tmp36
    tmp82 = tmp81 & tmp9
    tmp83 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp82, eviction_policy='evict_last', other=float("-inf"))
    tmp84 = triton_helpers.maximum(tmp83, tmp80)
    tmp85 = tmp29 & tmp44
    tmp86 = tmp85 & tmp9
    tmp87 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp86, eviction_policy='evict_last', other=float("-inf"))
    tmp88 = triton_helpers.maximum(tmp87, tmp84)
    tmp89 = tmp36 & tmp15
    tmp90 = tmp89 & tmp9
    tmp91 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp90, eviction_policy='evict_last', other=float("-inf"))
    tmp92 = triton_helpers.maximum(tmp91, tmp88)
    tmp93 = tmp36 & tmp22
    tmp94 = tmp93 & tmp9
    tmp95 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp94, eviction_policy='evict_last', other=float("-inf"))
    tmp96 = triton_helpers.maximum(tmp95, tmp92)
    tmp97 = tmp36 & tmp29
    tmp98 = tmp97 & tmp9
    tmp99 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp98, eviction_policy='evict_last', other=float("-inf"))
    tmp100 = triton_helpers.maximum(tmp99, tmp96)
    tmp101 = tmp36 & tmp36
    tmp102 = tmp101 & tmp9
    tmp103 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp102, eviction_policy='evict_last', other=float("-inf"))
    tmp104 = triton_helpers.maximum(tmp103, tmp100)
    tmp105 = tmp36 & tmp44
    tmp106 = tmp105 & tmp9
    tmp107 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp106, eviction_policy='evict_last', other=float("-inf"))
    tmp108 = triton_helpers.maximum(tmp107, tmp104)
    tmp109 = tmp44 & tmp15
    tmp110 = tmp109 & tmp9
    tmp111 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp110, eviction_policy='evict_last', other=float("-inf"))
    tmp112 = triton_helpers.maximum(tmp111, tmp108)
    tmp113 = tmp44 & tmp22
    tmp114 = tmp113 & tmp9
    tmp115 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp114, eviction_policy='evict_last', other=float("-inf"))
    tmp116 = triton_helpers.maximum(tmp115, tmp112)
    tmp117 = tmp44 & tmp29
    tmp118 = tmp117 & tmp9
    tmp119 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp118, eviction_policy='evict_last', other=float("-inf"))
    tmp120 = triton_helpers.maximum(tmp119, tmp116)
    tmp121 = tmp44 & tmp36
    tmp122 = tmp121 & tmp9
    tmp123 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp122, eviction_policy='evict_last', other=float("-inf"))
    tmp124 = triton_helpers.maximum(tmp123, tmp120)
    tmp125 = tmp44 & tmp44
    tmp126 = tmp125 & tmp9
    tmp127 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp126, eviction_policy='evict_last', other=float("-inf"))
    tmp128 = triton_helpers.maximum(tmp127, tmp124)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp9, tmp128, tmp129)
    tmp131 = tmp0 >= tmp7
    tmp132 = tl.full([1], 6144, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr1 + (2048*x1 + ((-4096) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tmp0 >= tmp132
    tmp137 = tl.full([1], 8192, tl.int64)
    tmp138 = tmp0 < tmp137
    tmp139 = tl.load(in_ptr2 + (2048*x1 + ((-6144) + x0)), tmp136, eviction_policy='evict_last', other=0.0)
    tmp140 = tl.where(tmp134, tmp135, tmp139)
    tmp141 = tl.where(tmp9, tmp130, tmp140)
    tmp142 = tl.where(tmp4, tmp5, tmp141)
    tl.store(out_ptr0 + (x2), tmp142, None)
''', device_str='cuda')


# kernel path: inductor_cache/jc/cjcmn756ysehpj3xtju74kw2v6c6mpsjpmjwjfxs3cthmxjnjkev.py
# Topologically Sorted Source Nodes: [batch_norm_185], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_185 => add_455, mul_741, mul_742, sub_185
# Graph fragment:
#   %sub_185 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_185, %unsqueeze_1481), kwargs = {})
#   %mul_741 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_185, %unsqueeze_1483), kwargs = {})
#   %mul_742 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_741, %unsqueeze_1485), kwargs = {})
#   %add_455 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_742, %unsqueeze_1487), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
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


# kernel path: inductor_cache/em/cemy5y6te5r6lzopn5uwfer4vrvoegpf7zq4f2pbnrtw5gxedcci.py
# Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_16 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_839, %mul_743], 1), kwargs = {})
triton_poi_fused_cat_28 = async_compile.triton('triton_poi_fused_cat_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_28(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4096)
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 4096, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr1 + (2048*x1 + ((-2048) + x0)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_2, (256, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_3, (256, ), (1, ))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_7, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (512, ), (1, ))
    assert_size_stride(primals_12, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (256, ), (1, ))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_32, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_42, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (256, ), (1, ))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (256, ), (1, ))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_57, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_82, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, ), (1, ))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (256, ), (1, ))
    assert_size_stride(primals_121, (256, ), (1, ))
    assert_size_stride(primals_122, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_133, (256, ), (1, ))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_138, (256, ), (1, ))
    assert_size_stride(primals_139, (256, ), (1, ))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (256, ), (1, ))
    assert_size_stride(primals_142, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, ), (1, ))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_148, (1024, ), (1, ))
    assert_size_stride(primals_149, (1024, ), (1, ))
    assert_size_stride(primals_150, (1024, ), (1, ))
    assert_size_stride(primals_151, (1024, ), (1, ))
    assert_size_stride(primals_152, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_153, (512, ), (1, ))
    assert_size_stride(primals_154, (512, ), (1, ))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (512, ), (1, ))
    assert_size_stride(primals_157, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_158, (512, ), (1, ))
    assert_size_stride(primals_159, (512, ), (1, ))
    assert_size_stride(primals_160, (512, ), (1, ))
    assert_size_stride(primals_161, (512, ), (1, ))
    assert_size_stride(primals_162, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_163, (512, ), (1, ))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_168, (512, ), (1, ))
    assert_size_stride(primals_169, (512, ), (1, ))
    assert_size_stride(primals_170, (512, ), (1, ))
    assert_size_stride(primals_171, (512, ), (1, ))
    assert_size_stride(primals_172, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_174, (512, ), (1, ))
    assert_size_stride(primals_175, (512, ), (1, ))
    assert_size_stride(primals_176, (512, ), (1, ))
    assert_size_stride(primals_177, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (512, ), (1, ))
    assert_size_stride(primals_182, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_183, (512, ), (1, ))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (512, ), (1, ))
    assert_size_stride(primals_186, (512, ), (1, ))
    assert_size_stride(primals_187, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_188, (512, ), (1, ))
    assert_size_stride(primals_189, (512, ), (1, ))
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_192, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_193, (512, ), (1, ))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_198, (512, ), (1, ))
    assert_size_stride(primals_199, (512, ), (1, ))
    assert_size_stride(primals_200, (512, ), (1, ))
    assert_size_stride(primals_201, (512, ), (1, ))
    assert_size_stride(primals_202, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_207, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_208, (512, ), (1, ))
    assert_size_stride(primals_209, (512, ), (1, ))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (512, ), (1, ))
    assert_size_stride(primals_212, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_216, (512, ), (1, ))
    assert_size_stride(primals_217, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_218, (512, ), (1, ))
    assert_size_stride(primals_219, (512, ), (1, ))
    assert_size_stride(primals_220, (512, ), (1, ))
    assert_size_stride(primals_221, (512, ), (1, ))
    assert_size_stride(primals_222, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (512, ), (1, ))
    assert_size_stride(primals_225, (512, ), (1, ))
    assert_size_stride(primals_226, (512, ), (1, ))
    assert_size_stride(primals_227, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_228, (512, ), (1, ))
    assert_size_stride(primals_229, (512, ), (1, ))
    assert_size_stride(primals_230, (512, ), (1, ))
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_233, (512, ), (1, ))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (512, ), (1, ))
    assert_size_stride(primals_236, (512, ), (1, ))
    assert_size_stride(primals_237, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_238, (512, ), (1, ))
    assert_size_stride(primals_239, (512, ), (1, ))
    assert_size_stride(primals_240, (512, ), (1, ))
    assert_size_stride(primals_241, (512, ), (1, ))
    assert_size_stride(primals_242, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_243, (512, ), (1, ))
    assert_size_stride(primals_244, (512, ), (1, ))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_248, (512, ), (1, ))
    assert_size_stride(primals_249, (512, ), (1, ))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_251, (512, ), (1, ))
    assert_size_stride(primals_252, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_253, (512, ), (1, ))
    assert_size_stride(primals_254, (512, ), (1, ))
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (512, ), (1, ))
    assert_size_stride(primals_260, (512, ), (1, ))
    assert_size_stride(primals_261, (512, ), (1, ))
    assert_size_stride(primals_262, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_263, (512, ), (1, ))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (512, ), (1, ))
    assert_size_stride(primals_266, (512, ), (1, ))
    assert_size_stride(primals_267, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (512, ), (1, ))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (512, ), (1, ))
    assert_size_stride(primals_272, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_273, (512, ), (1, ))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_275, (512, ), (1, ))
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_278, (512, ), (1, ))
    assert_size_stride(primals_279, (512, ), (1, ))
    assert_size_stride(primals_280, (512, ), (1, ))
    assert_size_stride(primals_281, (512, ), (1, ))
    assert_size_stride(primals_282, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_283, (512, ), (1, ))
    assert_size_stride(primals_284, (512, ), (1, ))
    assert_size_stride(primals_285, (512, ), (1, ))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_287, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (512, ), (1, ))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_291, (512, ), (1, ))
    assert_size_stride(primals_292, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (512, ), (1, ))
    assert_size_stride(primals_297, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_298, (512, ), (1, ))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_300, (512, ), (1, ))
    assert_size_stride(primals_301, (512, ), (1, ))
    assert_size_stride(primals_302, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_303, (512, ), (1, ))
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (512, 512, 3, 3), (4608, 9, 3, 1))
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
    assert_size_stride(primals_357, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_358, (512, ), (1, ))
    assert_size_stride(primals_359, (512, ), (1, ))
    assert_size_stride(primals_360, (512, ), (1, ))
    assert_size_stride(primals_361, (512, ), (1, ))
    assert_size_stride(primals_362, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_363, (512, ), (1, ))
    assert_size_stride(primals_364, (512, ), (1, ))
    assert_size_stride(primals_365, (512, ), (1, ))
    assert_size_stride(primals_366, (512, ), (1, ))
    assert_size_stride(primals_367, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_368, (512, ), (1, ))
    assert_size_stride(primals_369, (512, ), (1, ))
    assert_size_stride(primals_370, (512, ), (1, ))
    assert_size_stride(primals_371, (512, ), (1, ))
    assert_size_stride(primals_372, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_373, (512, ), (1, ))
    assert_size_stride(primals_374, (512, ), (1, ))
    assert_size_stride(primals_375, (512, ), (1, ))
    assert_size_stride(primals_376, (512, ), (1, ))
    assert_size_stride(primals_377, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_378, (512, ), (1, ))
    assert_size_stride(primals_379, (512, ), (1, ))
    assert_size_stride(primals_380, (512, ), (1, ))
    assert_size_stride(primals_381, (512, ), (1, ))
    assert_size_stride(primals_382, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_383, (512, ), (1, ))
    assert_size_stride(primals_384, (512, ), (1, ))
    assert_size_stride(primals_385, (512, ), (1, ))
    assert_size_stride(primals_386, (512, ), (1, ))
    assert_size_stride(primals_387, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_388, (512, ), (1, ))
    assert_size_stride(primals_389, (512, ), (1, ))
    assert_size_stride(primals_390, (512, ), (1, ))
    assert_size_stride(primals_391, (512, ), (1, ))
    assert_size_stride(primals_392, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_393, (512, ), (1, ))
    assert_size_stride(primals_394, (512, ), (1, ))
    assert_size_stride(primals_395, (512, ), (1, ))
    assert_size_stride(primals_396, (512, ), (1, ))
    assert_size_stride(primals_397, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_398, (512, ), (1, ))
    assert_size_stride(primals_399, (512, ), (1, ))
    assert_size_stride(primals_400, (512, ), (1, ))
    assert_size_stride(primals_401, (512, ), (1, ))
    assert_size_stride(primals_402, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_403, (512, ), (1, ))
    assert_size_stride(primals_404, (512, ), (1, ))
    assert_size_stride(primals_405, (512, ), (1, ))
    assert_size_stride(primals_406, (512, ), (1, ))
    assert_size_stride(primals_407, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_408, (512, ), (1, ))
    assert_size_stride(primals_409, (512, ), (1, ))
    assert_size_stride(primals_410, (512, ), (1, ))
    assert_size_stride(primals_411, (512, ), (1, ))
    assert_size_stride(primals_412, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_413, (512, ), (1, ))
    assert_size_stride(primals_414, (512, ), (1, ))
    assert_size_stride(primals_415, (512, ), (1, ))
    assert_size_stride(primals_416, (512, ), (1, ))
    assert_size_stride(primals_417, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_418, (512, ), (1, ))
    assert_size_stride(primals_419, (512, ), (1, ))
    assert_size_stride(primals_420, (512, ), (1, ))
    assert_size_stride(primals_421, (512, ), (1, ))
    assert_size_stride(primals_422, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_423, (512, ), (1, ))
    assert_size_stride(primals_424, (512, ), (1, ))
    assert_size_stride(primals_425, (512, ), (1, ))
    assert_size_stride(primals_426, (512, ), (1, ))
    assert_size_stride(primals_427, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_428, (512, ), (1, ))
    assert_size_stride(primals_429, (512, ), (1, ))
    assert_size_stride(primals_430, (512, ), (1, ))
    assert_size_stride(primals_431, (512, ), (1, ))
    assert_size_stride(primals_432, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_433, (512, ), (1, ))
    assert_size_stride(primals_434, (512, ), (1, ))
    assert_size_stride(primals_435, (512, ), (1, ))
    assert_size_stride(primals_436, (512, ), (1, ))
    assert_size_stride(primals_437, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_438, (512, ), (1, ))
    assert_size_stride(primals_439, (512, ), (1, ))
    assert_size_stride(primals_440, (512, ), (1, ))
    assert_size_stride(primals_441, (512, ), (1, ))
    assert_size_stride(primals_442, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_443, (512, ), (1, ))
    assert_size_stride(primals_444, (512, ), (1, ))
    assert_size_stride(primals_445, (512, ), (1, ))
    assert_size_stride(primals_446, (512, ), (1, ))
    assert_size_stride(primals_447, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_448, (512, ), (1, ))
    assert_size_stride(primals_449, (512, ), (1, ))
    assert_size_stride(primals_450, (512, ), (1, ))
    assert_size_stride(primals_451, (512, ), (1, ))
    assert_size_stride(primals_452, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_453, (512, ), (1, ))
    assert_size_stride(primals_454, (512, ), (1, ))
    assert_size_stride(primals_455, (512, ), (1, ))
    assert_size_stride(primals_456, (512, ), (1, ))
    assert_size_stride(primals_457, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_458, (512, ), (1, ))
    assert_size_stride(primals_459, (512, ), (1, ))
    assert_size_stride(primals_460, (512, ), (1, ))
    assert_size_stride(primals_461, (512, ), (1, ))
    assert_size_stride(primals_462, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_463, (512, ), (1, ))
    assert_size_stride(primals_464, (512, ), (1, ))
    assert_size_stride(primals_465, (512, ), (1, ))
    assert_size_stride(primals_466, (512, ), (1, ))
    assert_size_stride(primals_467, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_468, (512, ), (1, ))
    assert_size_stride(primals_469, (512, ), (1, ))
    assert_size_stride(primals_470, (512, ), (1, ))
    assert_size_stride(primals_471, (512, ), (1, ))
    assert_size_stride(primals_472, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_473, (512, ), (1, ))
    assert_size_stride(primals_474, (512, ), (1, ))
    assert_size_stride(primals_475, (512, ), (1, ))
    assert_size_stride(primals_476, (512, ), (1, ))
    assert_size_stride(primals_477, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_478, (512, ), (1, ))
    assert_size_stride(primals_479, (512, ), (1, ))
    assert_size_stride(primals_480, (512, ), (1, ))
    assert_size_stride(primals_481, (512, ), (1, ))
    assert_size_stride(primals_482, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_483, (512, ), (1, ))
    assert_size_stride(primals_484, (512, ), (1, ))
    assert_size_stride(primals_485, (512, ), (1, ))
    assert_size_stride(primals_486, (512, ), (1, ))
    assert_size_stride(primals_487, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_488, (512, ), (1, ))
    assert_size_stride(primals_489, (512, ), (1, ))
    assert_size_stride(primals_490, (512, ), (1, ))
    assert_size_stride(primals_491, (512, ), (1, ))
    assert_size_stride(primals_492, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_493, (512, ), (1, ))
    assert_size_stride(primals_494, (512, ), (1, ))
    assert_size_stride(primals_495, (512, ), (1, ))
    assert_size_stride(primals_496, (512, ), (1, ))
    assert_size_stride(primals_497, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_498, (512, ), (1, ))
    assert_size_stride(primals_499, (512, ), (1, ))
    assert_size_stride(primals_500, (512, ), (1, ))
    assert_size_stride(primals_501, (512, ), (1, ))
    assert_size_stride(primals_502, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_503, (512, ), (1, ))
    assert_size_stride(primals_504, (512, ), (1, ))
    assert_size_stride(primals_505, (512, ), (1, ))
    assert_size_stride(primals_506, (512, ), (1, ))
    assert_size_stride(primals_507, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_508, (512, ), (1, ))
    assert_size_stride(primals_509, (512, ), (1, ))
    assert_size_stride(primals_510, (512, ), (1, ))
    assert_size_stride(primals_511, (512, ), (1, ))
    assert_size_stride(primals_512, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_513, (512, ), (1, ))
    assert_size_stride(primals_514, (512, ), (1, ))
    assert_size_stride(primals_515, (512, ), (1, ))
    assert_size_stride(primals_516, (512, ), (1, ))
    assert_size_stride(primals_517, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_518, (512, ), (1, ))
    assert_size_stride(primals_519, (512, ), (1, ))
    assert_size_stride(primals_520, (512, ), (1, ))
    assert_size_stride(primals_521, (512, ), (1, ))
    assert_size_stride(primals_522, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_523, (1024, ), (1, ))
    assert_size_stride(primals_524, (1024, ), (1, ))
    assert_size_stride(primals_525, (1024, ), (1, ))
    assert_size_stride(primals_526, (1024, ), (1, ))
    assert_size_stride(primals_527, (2048, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_528, (2048, ), (1, ))
    assert_size_stride(primals_529, (2048, ), (1, ))
    assert_size_stride(primals_530, (2048, ), (1, ))
    assert_size_stride(primals_531, (2048, ), (1, ))
    assert_size_stride(primals_532, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_533, (1024, ), (1, ))
    assert_size_stride(primals_534, (1024, ), (1, ))
    assert_size_stride(primals_535, (1024, ), (1, ))
    assert_size_stride(primals_536, (1024, ), (1, ))
    assert_size_stride(primals_537, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_538, (1024, ), (1, ))
    assert_size_stride(primals_539, (1024, ), (1, ))
    assert_size_stride(primals_540, (1024, ), (1, ))
    assert_size_stride(primals_541, (1024, ), (1, ))
    assert_size_stride(primals_542, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_543, (1024, ), (1, ))
    assert_size_stride(primals_544, (1024, ), (1, ))
    assert_size_stride(primals_545, (1024, ), (1, ))
    assert_size_stride(primals_546, (1024, ), (1, ))
    assert_size_stride(primals_547, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_548, (1024, ), (1, ))
    assert_size_stride(primals_549, (1024, ), (1, ))
    assert_size_stride(primals_550, (1024, ), (1, ))
    assert_size_stride(primals_551, (1024, ), (1, ))
    assert_size_stride(primals_552, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_553, (1024, ), (1, ))
    assert_size_stride(primals_554, (1024, ), (1, ))
    assert_size_stride(primals_555, (1024, ), (1, ))
    assert_size_stride(primals_556, (1024, ), (1, ))
    assert_size_stride(primals_557, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_558, (1024, ), (1, ))
    assert_size_stride(primals_559, (1024, ), (1, ))
    assert_size_stride(primals_560, (1024, ), (1, ))
    assert_size_stride(primals_561, (1024, ), (1, ))
    assert_size_stride(primals_562, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_563, (1024, ), (1, ))
    assert_size_stride(primals_564, (1024, ), (1, ))
    assert_size_stride(primals_565, (1024, ), (1, ))
    assert_size_stride(primals_566, (1024, ), (1, ))
    assert_size_stride(primals_567, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_568, (1024, ), (1, ))
    assert_size_stride(primals_569, (1024, ), (1, ))
    assert_size_stride(primals_570, (1024, ), (1, ))
    assert_size_stride(primals_571, (1024, ), (1, ))
    assert_size_stride(primals_572, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_573, (1024, ), (1, ))
    assert_size_stride(primals_574, (1024, ), (1, ))
    assert_size_stride(primals_575, (1024, ), (1, ))
    assert_size_stride(primals_576, (1024, ), (1, ))
    assert_size_stride(primals_577, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_578, (1024, ), (1, ))
    assert_size_stride(primals_579, (1024, ), (1, ))
    assert_size_stride(primals_580, (1024, ), (1, ))
    assert_size_stride(primals_581, (1024, ), (1, ))
    assert_size_stride(primals_582, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_583, (1024, ), (1, ))
    assert_size_stride(primals_584, (1024, ), (1, ))
    assert_size_stride(primals_585, (1024, ), (1, ))
    assert_size_stride(primals_586, (1024, ), (1, ))
    assert_size_stride(primals_587, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_588, (1024, ), (1, ))
    assert_size_stride(primals_589, (1024, ), (1, ))
    assert_size_stride(primals_590, (1024, ), (1, ))
    assert_size_stride(primals_591, (1024, ), (1, ))
    assert_size_stride(primals_592, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_593, (1024, ), (1, ))
    assert_size_stride(primals_594, (1024, ), (1, ))
    assert_size_stride(primals_595, (1024, ), (1, ))
    assert_size_stride(primals_596, (1024, ), (1, ))
    assert_size_stride(primals_597, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_598, (1024, ), (1, ))
    assert_size_stride(primals_599, (1024, ), (1, ))
    assert_size_stride(primals_600, (1024, ), (1, ))
    assert_size_stride(primals_601, (1024, ), (1, ))
    assert_size_stride(primals_602, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_603, (1024, ), (1, ))
    assert_size_stride(primals_604, (1024, ), (1, ))
    assert_size_stride(primals_605, (1024, ), (1, ))
    assert_size_stride(primals_606, (1024, ), (1, ))
    assert_size_stride(primals_607, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_608, (1024, ), (1, ))
    assert_size_stride(primals_609, (1024, ), (1, ))
    assert_size_stride(primals_610, (1024, ), (1, ))
    assert_size_stride(primals_611, (1024, ), (1, ))
    assert_size_stride(primals_612, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_613, (1024, ), (1, ))
    assert_size_stride(primals_614, (1024, ), (1, ))
    assert_size_stride(primals_615, (1024, ), (1, ))
    assert_size_stride(primals_616, (1024, ), (1, ))
    assert_size_stride(primals_617, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_618, (1024, ), (1, ))
    assert_size_stride(primals_619, (1024, ), (1, ))
    assert_size_stride(primals_620, (1024, ), (1, ))
    assert_size_stride(primals_621, (1024, ), (1, ))
    assert_size_stride(primals_622, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_623, (1024, ), (1, ))
    assert_size_stride(primals_624, (1024, ), (1, ))
    assert_size_stride(primals_625, (1024, ), (1, ))
    assert_size_stride(primals_626, (1024, ), (1, ))
    assert_size_stride(primals_627, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_628, (1024, ), (1, ))
    assert_size_stride(primals_629, (1024, ), (1, ))
    assert_size_stride(primals_630, (1024, ), (1, ))
    assert_size_stride(primals_631, (1024, ), (1, ))
    assert_size_stride(primals_632, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_633, (1024, ), (1, ))
    assert_size_stride(primals_634, (1024, ), (1, ))
    assert_size_stride(primals_635, (1024, ), (1, ))
    assert_size_stride(primals_636, (1024, ), (1, ))
    assert_size_stride(primals_637, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_638, (1024, ), (1, ))
    assert_size_stride(primals_639, (1024, ), (1, ))
    assert_size_stride(primals_640, (1024, ), (1, ))
    assert_size_stride(primals_641, (1024, ), (1, ))
    assert_size_stride(primals_642, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_643, (1024, ), (1, ))
    assert_size_stride(primals_644, (1024, ), (1, ))
    assert_size_stride(primals_645, (1024, ), (1, ))
    assert_size_stride(primals_646, (1024, ), (1, ))
    assert_size_stride(primals_647, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_648, (1024, ), (1, ))
    assert_size_stride(primals_649, (1024, ), (1, ))
    assert_size_stride(primals_650, (1024, ), (1, ))
    assert_size_stride(primals_651, (1024, ), (1, ))
    assert_size_stride(primals_652, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_653, (1024, ), (1, ))
    assert_size_stride(primals_654, (1024, ), (1, ))
    assert_size_stride(primals_655, (1024, ), (1, ))
    assert_size_stride(primals_656, (1024, ), (1, ))
    assert_size_stride(primals_657, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_658, (1024, ), (1, ))
    assert_size_stride(primals_659, (1024, ), (1, ))
    assert_size_stride(primals_660, (1024, ), (1, ))
    assert_size_stride(primals_661, (1024, ), (1, ))
    assert_size_stride(primals_662, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_663, (1024, ), (1, ))
    assert_size_stride(primals_664, (1024, ), (1, ))
    assert_size_stride(primals_665, (1024, ), (1, ))
    assert_size_stride(primals_666, (1024, ), (1, ))
    assert_size_stride(primals_667, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_668, (1024, ), (1, ))
    assert_size_stride(primals_669, (1024, ), (1, ))
    assert_size_stride(primals_670, (1024, ), (1, ))
    assert_size_stride(primals_671, (1024, ), (1, ))
    assert_size_stride(primals_672, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_673, (1024, ), (1, ))
    assert_size_stride(primals_674, (1024, ), (1, ))
    assert_size_stride(primals_675, (1024, ), (1, ))
    assert_size_stride(primals_676, (1024, ), (1, ))
    assert_size_stride(primals_677, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_678, (1024, ), (1, ))
    assert_size_stride(primals_679, (1024, ), (1, ))
    assert_size_stride(primals_680, (1024, ), (1, ))
    assert_size_stride(primals_681, (1024, ), (1, ))
    assert_size_stride(primals_682, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_683, (1024, ), (1, ))
    assert_size_stride(primals_684, (1024, ), (1, ))
    assert_size_stride(primals_685, (1024, ), (1, ))
    assert_size_stride(primals_686, (1024, ), (1, ))
    assert_size_stride(primals_687, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_688, (1024, ), (1, ))
    assert_size_stride(primals_689, (1024, ), (1, ))
    assert_size_stride(primals_690, (1024, ), (1, ))
    assert_size_stride(primals_691, (1024, ), (1, ))
    assert_size_stride(primals_692, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_693, (1024, ), (1, ))
    assert_size_stride(primals_694, (1024, ), (1, ))
    assert_size_stride(primals_695, (1024, ), (1, ))
    assert_size_stride(primals_696, (1024, ), (1, ))
    assert_size_stride(primals_697, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_698, (1024, ), (1, ))
    assert_size_stride(primals_699, (1024, ), (1, ))
    assert_size_stride(primals_700, (1024, ), (1, ))
    assert_size_stride(primals_701, (1024, ), (1, ))
    assert_size_stride(primals_702, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_703, (1024, ), (1, ))
    assert_size_stride(primals_704, (1024, ), (1, ))
    assert_size_stride(primals_705, (1024, ), (1, ))
    assert_size_stride(primals_706, (1024, ), (1, ))
    assert_size_stride(primals_707, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_708, (1024, ), (1, ))
    assert_size_stride(primals_709, (1024, ), (1, ))
    assert_size_stride(primals_710, (1024, ), (1, ))
    assert_size_stride(primals_711, (1024, ), (1, ))
    assert_size_stride(primals_712, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_713, (1024, ), (1, ))
    assert_size_stride(primals_714, (1024, ), (1, ))
    assert_size_stride(primals_715, (1024, ), (1, ))
    assert_size_stride(primals_716, (1024, ), (1, ))
    assert_size_stride(primals_717, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_718, (1024, ), (1, ))
    assert_size_stride(primals_719, (1024, ), (1, ))
    assert_size_stride(primals_720, (1024, ), (1, ))
    assert_size_stride(primals_721, (1024, ), (1, ))
    assert_size_stride(primals_722, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_723, (1024, ), (1, ))
    assert_size_stride(primals_724, (1024, ), (1, ))
    assert_size_stride(primals_725, (1024, ), (1, ))
    assert_size_stride(primals_726, (1024, ), (1, ))
    assert_size_stride(primals_727, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_728, (1024, ), (1, ))
    assert_size_stride(primals_729, (1024, ), (1, ))
    assert_size_stride(primals_730, (1024, ), (1, ))
    assert_size_stride(primals_731, (1024, ), (1, ))
    assert_size_stride(primals_732, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_733, (1024, ), (1, ))
    assert_size_stride(primals_734, (1024, ), (1, ))
    assert_size_stride(primals_735, (1024, ), (1, ))
    assert_size_stride(primals_736, (1024, ), (1, ))
    assert_size_stride(primals_737, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_738, (1024, ), (1, ))
    assert_size_stride(primals_739, (1024, ), (1, ))
    assert_size_stride(primals_740, (1024, ), (1, ))
    assert_size_stride(primals_741, (1024, ), (1, ))
    assert_size_stride(primals_742, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_743, (1024, ), (1, ))
    assert_size_stride(primals_744, (1024, ), (1, ))
    assert_size_stride(primals_745, (1024, ), (1, ))
    assert_size_stride(primals_746, (1024, ), (1, ))
    assert_size_stride(primals_747, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_748, (1024, ), (1, ))
    assert_size_stride(primals_749, (1024, ), (1, ))
    assert_size_stride(primals_750, (1024, ), (1, ))
    assert_size_stride(primals_751, (1024, ), (1, ))
    assert_size_stride(primals_752, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_753, (1024, ), (1, ))
    assert_size_stride(primals_754, (1024, ), (1, ))
    assert_size_stride(primals_755, (1024, ), (1, ))
    assert_size_stride(primals_756, (1024, ), (1, ))
    assert_size_stride(primals_757, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_758, (1024, ), (1, ))
    assert_size_stride(primals_759, (1024, ), (1, ))
    assert_size_stride(primals_760, (1024, ), (1, ))
    assert_size_stride(primals_761, (1024, ), (1, ))
    assert_size_stride(primals_762, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_763, (1024, ), (1, ))
    assert_size_stride(primals_764, (1024, ), (1, ))
    assert_size_stride(primals_765, (1024, ), (1, ))
    assert_size_stride(primals_766, (1024, ), (1, ))
    assert_size_stride(primals_767, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_768, (1024, ), (1, ))
    assert_size_stride(primals_769, (1024, ), (1, ))
    assert_size_stride(primals_770, (1024, ), (1, ))
    assert_size_stride(primals_771, (1024, ), (1, ))
    assert_size_stride(primals_772, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_773, (1024, ), (1, ))
    assert_size_stride(primals_774, (1024, ), (1, ))
    assert_size_stride(primals_775, (1024, ), (1, ))
    assert_size_stride(primals_776, (1024, ), (1, ))
    assert_size_stride(primals_777, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_778, (1024, ), (1, ))
    assert_size_stride(primals_779, (1024, ), (1, ))
    assert_size_stride(primals_780, (1024, ), (1, ))
    assert_size_stride(primals_781, (1024, ), (1, ))
    assert_size_stride(primals_782, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_783, (1024, ), (1, ))
    assert_size_stride(primals_784, (1024, ), (1, ))
    assert_size_stride(primals_785, (1024, ), (1, ))
    assert_size_stride(primals_786, (1024, ), (1, ))
    assert_size_stride(primals_787, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_788, (1024, ), (1, ))
    assert_size_stride(primals_789, (1024, ), (1, ))
    assert_size_stride(primals_790, (1024, ), (1, ))
    assert_size_stride(primals_791, (1024, ), (1, ))
    assert_size_stride(primals_792, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_793, (1024, ), (1, ))
    assert_size_stride(primals_794, (1024, ), (1, ))
    assert_size_stride(primals_795, (1024, ), (1, ))
    assert_size_stride(primals_796, (1024, ), (1, ))
    assert_size_stride(primals_797, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_798, (1024, ), (1, ))
    assert_size_stride(primals_799, (1024, ), (1, ))
    assert_size_stride(primals_800, (1024, ), (1, ))
    assert_size_stride(primals_801, (1024, ), (1, ))
    assert_size_stride(primals_802, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_803, (1024, ), (1, ))
    assert_size_stride(primals_804, (1024, ), (1, ))
    assert_size_stride(primals_805, (1024, ), (1, ))
    assert_size_stride(primals_806, (1024, ), (1, ))
    assert_size_stride(primals_807, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_808, (1024, ), (1, ))
    assert_size_stride(primals_809, (1024, ), (1, ))
    assert_size_stride(primals_810, (1024, ), (1, ))
    assert_size_stride(primals_811, (1024, ), (1, ))
    assert_size_stride(primals_812, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_813, (1024, ), (1, ))
    assert_size_stride(primals_814, (1024, ), (1, ))
    assert_size_stride(primals_815, (1024, ), (1, ))
    assert_size_stride(primals_816, (1024, ), (1, ))
    assert_size_stride(primals_817, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_818, (1024, ), (1, ))
    assert_size_stride(primals_819, (1024, ), (1, ))
    assert_size_stride(primals_820, (1024, ), (1, ))
    assert_size_stride(primals_821, (1024, ), (1, ))
    assert_size_stride(primals_822, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_823, (1024, ), (1, ))
    assert_size_stride(primals_824, (1024, ), (1, ))
    assert_size_stride(primals_825, (1024, ), (1, ))
    assert_size_stride(primals_826, (1024, ), (1, ))
    assert_size_stride(primals_827, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_828, (1024, ), (1, ))
    assert_size_stride(primals_829, (1024, ), (1, ))
    assert_size_stride(primals_830, (1024, ), (1, ))
    assert_size_stride(primals_831, (1024, ), (1, ))
    assert_size_stride(primals_832, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_833, (1024, ), (1, ))
    assert_size_stride(primals_834, (1024, ), (1, ))
    assert_size_stride(primals_835, (1024, ), (1, ))
    assert_size_stride(primals_836, (1024, ), (1, ))
    assert_size_stride(primals_837, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_838, (1024, ), (1, ))
    assert_size_stride(primals_839, (1024, ), (1, ))
    assert_size_stride(primals_840, (1024, ), (1, ))
    assert_size_stride(primals_841, (1024, ), (1, ))
    assert_size_stride(primals_842, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_843, (1024, ), (1, ))
    assert_size_stride(primals_844, (1024, ), (1, ))
    assert_size_stride(primals_845, (1024, ), (1, ))
    assert_size_stride(primals_846, (1024, ), (1, ))
    assert_size_stride(primals_847, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_848, (1024, ), (1, ))
    assert_size_stride(primals_849, (1024, ), (1, ))
    assert_size_stride(primals_850, (1024, ), (1, ))
    assert_size_stride(primals_851, (1024, ), (1, ))
    assert_size_stride(primals_852, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_853, (1024, ), (1, ))
    assert_size_stride(primals_854, (1024, ), (1, ))
    assert_size_stride(primals_855, (1024, ), (1, ))
    assert_size_stride(primals_856, (1024, ), (1, ))
    assert_size_stride(primals_857, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_858, (1024, ), (1, ))
    assert_size_stride(primals_859, (1024, ), (1, ))
    assert_size_stride(primals_860, (1024, ), (1, ))
    assert_size_stride(primals_861, (1024, ), (1, ))
    assert_size_stride(primals_862, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_863, (1024, ), (1, ))
    assert_size_stride(primals_864, (1024, ), (1, ))
    assert_size_stride(primals_865, (1024, ), (1, ))
    assert_size_stride(primals_866, (1024, ), (1, ))
    assert_size_stride(primals_867, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_868, (1024, ), (1, ))
    assert_size_stride(primals_869, (1024, ), (1, ))
    assert_size_stride(primals_870, (1024, ), (1, ))
    assert_size_stride(primals_871, (1024, ), (1, ))
    assert_size_stride(primals_872, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_873, (1024, ), (1, ))
    assert_size_stride(primals_874, (1024, ), (1, ))
    assert_size_stride(primals_875, (1024, ), (1, ))
    assert_size_stride(primals_876, (1024, ), (1, ))
    assert_size_stride(primals_877, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_878, (1024, ), (1, ))
    assert_size_stride(primals_879, (1024, ), (1, ))
    assert_size_stride(primals_880, (1024, ), (1, ))
    assert_size_stride(primals_881, (1024, ), (1, ))
    assert_size_stride(primals_882, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_883, (1024, ), (1, ))
    assert_size_stride(primals_884, (1024, ), (1, ))
    assert_size_stride(primals_885, (1024, ), (1, ))
    assert_size_stride(primals_886, (1024, ), (1, ))
    assert_size_stride(primals_887, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_888, (1024, ), (1, ))
    assert_size_stride(primals_889, (1024, ), (1, ))
    assert_size_stride(primals_890, (1024, ), (1, ))
    assert_size_stride(primals_891, (1024, ), (1, ))
    assert_size_stride(primals_892, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_893, (1024, ), (1, ))
    assert_size_stride(primals_894, (1024, ), (1, ))
    assert_size_stride(primals_895, (1024, ), (1, ))
    assert_size_stride(primals_896, (1024, ), (1, ))
    assert_size_stride(primals_897, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_898, (1024, ), (1, ))
    assert_size_stride(primals_899, (1024, ), (1, ))
    assert_size_stride(primals_900, (1024, ), (1, ))
    assert_size_stride(primals_901, (1024, ), (1, ))
    assert_size_stride(primals_902, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_903, (2048, ), (1, ))
    assert_size_stride(primals_904, (2048, ), (1, ))
    assert_size_stride(primals_905, (2048, ), (1, ))
    assert_size_stride(primals_906, (2048, ), (1, ))
    assert_size_stride(primals_907, (4096, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_908, (4096, ), (1, ))
    assert_size_stride(primals_909, (4096, ), (1, ))
    assert_size_stride(primals_910, (4096, ), (1, ))
    assert_size_stride(primals_911, (4096, ), (1, ))
    assert_size_stride(primals_912, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_913, (2048, ), (1, ))
    assert_size_stride(primals_914, (2048, ), (1, ))
    assert_size_stride(primals_915, (2048, ), (1, ))
    assert_size_stride(primals_916, (2048, ), (1, ))
    assert_size_stride(primals_917, (4096, 8192, 1, 1), (8192, 1, 1, 1))
    assert_size_stride(primals_918, (4096, ), (1, ))
    assert_size_stride(primals_919, (4096, ), (1, ))
    assert_size_stride(primals_920, (4096, ), (1, ))
    assert_size_stride(primals_921, (4096, ), (1, ))
    assert_size_stride(primals_922, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_923, (2048, ), (1, ))
    assert_size_stride(primals_924, (2048, ), (1, ))
    assert_size_stride(primals_925, (2048, ), (1, ))
    assert_size_stride(primals_926, (2048, ), (1, ))
    assert_size_stride(primals_927, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_928, (2048, ), (1, ))
    assert_size_stride(primals_929, (2048, ), (1, ))
    assert_size_stride(primals_930, (2048, ), (1, ))
    assert_size_stride(primals_931, (2048, ), (1, ))
    assert_size_stride(primals_932, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_933, (2048, ), (1, ))
    assert_size_stride(primals_934, (2048, ), (1, ))
    assert_size_stride(primals_935, (2048, ), (1, ))
    assert_size_stride(primals_936, (2048, ), (1, ))
    assert_size_stride(primals_937, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_938, (2048, ), (1, ))
    assert_size_stride(primals_939, (2048, ), (1, ))
    assert_size_stride(primals_940, (2048, ), (1, ))
    assert_size_stride(primals_941, (2048, ), (1, ))
    assert_size_stride(primals_942, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_943, (2048, ), (1, ))
    assert_size_stride(primals_944, (2048, ), (1, ))
    assert_size_stride(primals_945, (2048, ), (1, ))
    assert_size_stride(primals_946, (2048, ), (1, ))
    assert_size_stride(primals_947, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_948, (2048, ), (1, ))
    assert_size_stride(primals_949, (2048, ), (1, ))
    assert_size_stride(primals_950, (2048, ), (1, ))
    assert_size_stride(primals_951, (2048, ), (1, ))
    assert_size_stride(primals_952, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_953, (2048, ), (1, ))
    assert_size_stride(primals_954, (2048, ), (1, ))
    assert_size_stride(primals_955, (2048, ), (1, ))
    assert_size_stride(primals_956, (2048, ), (1, ))
    assert_size_stride(primals_957, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_958, (2048, ), (1, ))
    assert_size_stride(primals_959, (2048, ), (1, ))
    assert_size_stride(primals_960, (2048, ), (1, ))
    assert_size_stride(primals_961, (2048, ), (1, ))
    assert_size_stride(primals_962, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_963, (2048, ), (1, ))
    assert_size_stride(primals_964, (2048, ), (1, ))
    assert_size_stride(primals_965, (2048, ), (1, ))
    assert_size_stride(primals_966, (2048, ), (1, ))
    assert_size_stride(primals_967, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_968, (2048, ), (1, ))
    assert_size_stride(primals_969, (2048, ), (1, ))
    assert_size_stride(primals_970, (2048, ), (1, ))
    assert_size_stride(primals_971, (2048, ), (1, ))
    assert_size_stride(primals_972, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_973, (2048, ), (1, ))
    assert_size_stride(primals_974, (2048, ), (1, ))
    assert_size_stride(primals_975, (2048, ), (1, ))
    assert_size_stride(primals_976, (2048, ), (1, ))
    assert_size_stride(primals_977, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_978, (2048, ), (1, ))
    assert_size_stride(primals_979, (2048, ), (1, ))
    assert_size_stride(primals_980, (2048, ), (1, ))
    assert_size_stride(primals_981, (2048, ), (1, ))
    assert_size_stride(primals_982, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_983, (2048, ), (1, ))
    assert_size_stride(primals_984, (2048, ), (1, ))
    assert_size_stride(primals_985, (2048, ), (1, ))
    assert_size_stride(primals_986, (2048, ), (1, ))
    assert_size_stride(primals_987, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_988, (2048, ), (1, ))
    assert_size_stride(primals_989, (2048, ), (1, ))
    assert_size_stride(primals_990, (2048, ), (1, ))
    assert_size_stride(primals_991, (2048, ), (1, ))
    assert_size_stride(primals_992, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_993, (2048, ), (1, ))
    assert_size_stride(primals_994, (2048, ), (1, ))
    assert_size_stride(primals_995, (2048, ), (1, ))
    assert_size_stride(primals_996, (2048, ), (1, ))
    assert_size_stride(primals_997, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_998, (2048, ), (1, ))
    assert_size_stride(primals_999, (2048, ), (1, ))
    assert_size_stride(primals_1000, (2048, ), (1, ))
    assert_size_stride(primals_1001, (2048, ), (1, ))
    assert_size_stride(primals_1002, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_1003, (2048, ), (1, ))
    assert_size_stride(primals_1004, (2048, ), (1, ))
    assert_size_stride(primals_1005, (2048, ), (1, ))
    assert_size_stride(primals_1006, (2048, ), (1, ))
    assert_size_stride(primals_1007, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_1008, (2048, ), (1, ))
    assert_size_stride(primals_1009, (2048, ), (1, ))
    assert_size_stride(primals_1010, (2048, ), (1, ))
    assert_size_stride(primals_1011, (2048, ), (1, ))
    assert_size_stride(primals_1012, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_1013, (2048, ), (1, ))
    assert_size_stride(primals_1014, (2048, ), (1, ))
    assert_size_stride(primals_1015, (2048, ), (1, ))
    assert_size_stride(primals_1016, (2048, ), (1, ))
    assert_size_stride(primals_1017, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_1018, (2048, ), (1, ))
    assert_size_stride(primals_1019, (2048, ), (1, ))
    assert_size_stride(primals_1020, (2048, ), (1, ))
    assert_size_stride(primals_1021, (2048, ), (1, ))
    assert_size_stride(primals_1022, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_1023, (2048, ), (1, ))
    assert_size_stride(primals_1024, (2048, ), (1, ))
    assert_size_stride(primals_1025, (2048, ), (1, ))
    assert_size_stride(primals_1026, (2048, ), (1, ))
    assert_size_stride(primals_1027, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_1028, (2048, ), (1, ))
    assert_size_stride(primals_1029, (2048, ), (1, ))
    assert_size_stride(primals_1030, (2048, ), (1, ))
    assert_size_stride(primals_1031, (2048, ), (1, ))
    assert_size_stride(primals_1032, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_1033, (2048, ), (1, ))
    assert_size_stride(primals_1034, (2048, ), (1, ))
    assert_size_stride(primals_1035, (2048, ), (1, ))
    assert_size_stride(primals_1036, (2048, ), (1, ))
    assert_size_stride(primals_1037, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_1038, (2048, ), (1, ))
    assert_size_stride(primals_1039, (2048, ), (1, ))
    assert_size_stride(primals_1040, (2048, ), (1, ))
    assert_size_stride(primals_1041, (2048, ), (1, ))
    assert_size_stride(primals_1042, (2048, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_1043, (2048, ), (1, ))
    assert_size_stride(primals_1044, (2048, ), (1, ))
    assert_size_stride(primals_1045, (2048, ), (1, ))
    assert_size_stride(primals_1046, (2048, ), (1, ))
    assert_size_stride(primals_1047, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_1048, (2048, ), (1, ))
    assert_size_stride(primals_1049, (2048, ), (1, ))
    assert_size_stride(primals_1050, (2048, ), (1, ))
    assert_size_stride(primals_1051, (2048, ), (1, ))
    assert_size_stride(primals_1052, (4096, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_1053, (4096, ), (1, ))
    assert_size_stride(primals_1054, (4096, ), (1, ))
    assert_size_stride(primals_1055, (4096, ), (1, ))
    assert_size_stride(primals_1056, (4096, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 12, 3, 3), (108, 1, 36, 12), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_2, buf0, 3072, 9, grid=grid(3072, 9), stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_7, buf1, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_7
        buf2 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_27, buf2, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_27
        buf3 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_37, buf3, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_37
        buf4 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_47, buf4, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_47
        buf5 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_57, buf5, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_57
        buf6 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_67, buf6, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_67
        buf7 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_77, buf7, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_77
        buf8 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_87, buf8, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_87
        buf9 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_97, buf9, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_97
        buf10 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_107, buf10, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_107
        buf11 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_117, buf11, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_117
        buf12 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_127, buf12, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_127
        buf13 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_137, buf13, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_137
        buf14 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_147, buf14, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_147
        buf15 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_167, buf15, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_167
        buf16 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_177, buf16, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_177
        buf17 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_187, buf17, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_187
        buf18 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_197, buf18, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_197
        buf19 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_207, buf19, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_207
        buf20 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_217, buf20, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_217
        buf21 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_227, buf21, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_227
        buf22 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_237, buf22, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_237
        buf23 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_247, buf23, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_247
        buf24 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_257, buf24, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_257
        buf25 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_267, buf25, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_267
        buf26 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_277, buf26, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_277
        buf27 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_287, buf27, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_287
        buf28 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_297, buf28, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_297
        buf29 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_307, buf29, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_307
        buf30 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_317, buf30, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_317
        buf31 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_327, buf31, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_327
        buf32 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_337, buf32, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_337
        buf33 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_347, buf33, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_347
        buf34 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_357, buf34, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_357
        buf35 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_367, buf35, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_367
        buf36 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_377, buf36, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_377
        buf37 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_387, buf37, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_387
        buf38 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_397, buf38, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_397
        buf39 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_407, buf39, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_407
        buf40 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_417, buf40, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_417
        buf41 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_427, buf41, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_427
        buf42 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_437, buf42, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_437
        buf43 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_447, buf43, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_447
        buf44 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_457, buf44, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_457
        buf45 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_467, buf45, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_467
        buf46 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_477, buf46, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_477
        buf47 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_487, buf47, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_487
        buf48 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_497, buf48, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_497
        buf49 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_507, buf49, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_507
        buf50 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_517, buf50, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_517
        buf51 = empty_strided_cuda((2048, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_527, buf51, 2097152, 9, grid=grid(2097152, 9), stream=stream0)
        del primals_527
        buf52 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_547, buf52, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_547
        buf53 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_557, buf53, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_557
        buf54 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_567, buf54, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_567
        buf55 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_577, buf55, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_577
        buf56 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_587, buf56, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_587
        buf57 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_597, buf57, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_597
        buf58 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_607, buf58, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_607
        buf59 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_617, buf59, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_617
        buf60 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_627, buf60, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_627
        buf61 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_637, buf61, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_637
        buf62 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_647, buf62, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_647
        buf63 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_657, buf63, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_657
        buf64 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_667, buf64, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_667
        buf65 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_677, buf65, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_677
        buf66 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_687, buf66, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_687
        buf67 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_697, buf67, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_697
        buf68 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_707, buf68, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_707
        buf69 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_717, buf69, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_717
        buf70 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_727, buf70, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_727
        buf71 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_737, buf71, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_737
        buf72 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_747, buf72, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_747
        buf73 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_757, buf73, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_757
        buf74 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_767, buf74, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_767
        buf75 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_777, buf75, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_777
        buf76 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_787, buf76, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_787
        buf77 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_797, buf77, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_797
        buf78 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_807, buf78, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_807
        buf79 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_817, buf79, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_817
        buf80 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_827, buf80, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_827
        buf81 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_837, buf81, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_837
        buf82 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_847, buf82, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_847
        buf83 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_857, buf83, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_857
        buf84 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_867, buf84, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_867
        buf85 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_877, buf85, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_877
        buf86 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_887, buf86, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_887
        buf87 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_897, buf87, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_897
        buf88 = empty_strided_cuda((4096, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_907, buf88, 8388608, 9, grid=grid(8388608, 9), stream=stream0)
        del primals_907
        buf89 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_937, buf89, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_937
        buf90 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_947, buf90, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_947
        buf91 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_957, buf91, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_957
        buf92 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_967, buf92, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_967
        buf93 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_977, buf93, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_977
        buf94 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_987, buf94, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_987
        buf95 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_997, buf95, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_997
        buf96 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_1007, buf96, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_1007
        buf97 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_1017, buf97, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_1017
        buf98 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_1027, buf98, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_1027
        buf99 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_1037, buf99, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_1037
        buf100 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_1047, buf100, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_1047
        buf101 = empty_strided_cuda((4, 12, 2, 2), (48, 1, 24, 12), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(primals_1, buf101, 192, grid=grid(192), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf103 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [batch_norm, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_10.run(buf104, buf102, primals_3, primals_4, primals_5, primals_6, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 512, 1, 1), (512, 1, 512, 512))
        buf106 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf107 = reinterpret_tensor(buf106, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_1, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf107, buf105, primals_8, primals_9, primals_10, primals_11, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 256, 1, 1), (256, 1, 256, 256))
        buf109 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf110 = reinterpret_tensor(buf109, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_2, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf110, buf108, primals_13, primals_14, primals_15, primals_16, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf107, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 256, 1, 1), (256, 1, 256, 256))
        buf112 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf111, primals_18, primals_19, primals_20, primals_21, buf112, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf110, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 256, 1, 1), (256, 1, 256, 256))
        buf114 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf115 = reinterpret_tensor(buf114, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_4, silu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf115, buf113, primals_23, primals_24, primals_25, primals_26, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 256, 1, 1), (256, 1, 256, 256))
        buf117 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf118 = reinterpret_tensor(buf117, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_5, y, y_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf118, buf116, primals_28, primals_29, primals_30, primals_31, buf110, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 256, 1, 1), (256, 1, 256, 256))
        buf120 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf121 = reinterpret_tensor(buf120, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_6, silu_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf121, buf119, primals_33, primals_34, primals_35, primals_36, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 256, 1, 1), (256, 1, 256, 256))
        buf123 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf124 = reinterpret_tensor(buf123, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_7, y_2, y_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf124, buf122, primals_38, primals_39, primals_40, primals_41, buf118, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 256, 1, 1), (256, 1, 256, 256))
        buf126 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf127 = reinterpret_tensor(buf126, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_8, silu_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf127, buf125, primals_43, primals_44, primals_45, primals_46, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 256, 1, 1), (256, 1, 256, 256))
        buf129 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf130 = reinterpret_tensor(buf129, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_9, y_4, y_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf130, buf128, primals_48, primals_49, primals_50, primals_51, buf124, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 256, 1, 1), (256, 1, 256, 256))
        buf132 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf133 = reinterpret_tensor(buf132, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_10, silu_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf133, buf131, primals_53, primals_54, primals_55, primals_56, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 256, 1, 1), (256, 1, 256, 256))
        buf135 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf136 = reinterpret_tensor(buf135, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_11, y_6, y_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf136, buf134, primals_58, primals_59, primals_60, primals_61, buf130, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 256, 1, 1), (256, 1, 256, 256))
        buf138 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf139 = reinterpret_tensor(buf138, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_12, silu_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf139, buf137, primals_63, primals_64, primals_65, primals_66, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 256, 1, 1), (256, 1, 256, 256))
        buf141 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf142 = reinterpret_tensor(buf141, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_13, y_8, y_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf142, buf140, primals_68, primals_69, primals_70, primals_71, buf136, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 256, 1, 1), (256, 1, 256, 256))
        buf144 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf145 = reinterpret_tensor(buf144, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_14, silu_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf145, buf143, primals_73, primals_74, primals_75, primals_76, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 256, 1, 1), (256, 1, 256, 256))
        buf147 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf148 = reinterpret_tensor(buf147, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_15, y_10, y_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf148, buf146, primals_78, primals_79, primals_80, primals_81, buf142, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 256, 1, 1), (256, 1, 256, 256))
        buf150 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf151 = reinterpret_tensor(buf150, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_16, silu_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf151, buf149, primals_83, primals_84, primals_85, primals_86, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 256, 1, 1), (256, 1, 256, 256))
        buf153 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf154 = reinterpret_tensor(buf153, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_17, y_12, y_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf154, buf152, primals_88, primals_89, primals_90, primals_91, buf148, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 256, 1, 1), (256, 1, 256, 256))
        buf156 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf157 = reinterpret_tensor(buf156, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_18, silu_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf157, buf155, primals_93, primals_94, primals_95, primals_96, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 256, 1, 1), (256, 1, 256, 256))
        buf159 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf160 = reinterpret_tensor(buf159, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_19, y_14, y_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf160, buf158, primals_98, primals_99, primals_100, primals_101, buf154, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 256, 1, 1), (256, 1, 256, 256))
        buf162 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf163 = reinterpret_tensor(buf162, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_20, silu_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf163, buf161, primals_103, primals_104, primals_105, primals_106, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 256, 1, 1), (256, 1, 256, 256))
        buf165 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf166 = reinterpret_tensor(buf165, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_21, y_16, y_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf166, buf164, primals_108, primals_109, primals_110, primals_111, buf160, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 256, 1, 1), (256, 1, 256, 256))
        buf168 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf169 = reinterpret_tensor(buf168, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_22, silu_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf169, buf167, primals_113, primals_114, primals_115, primals_116, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 256, 1, 1), (256, 1, 256, 256))
        buf171 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf172 = reinterpret_tensor(buf171, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_23, y_18, y_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf172, buf170, primals_118, primals_119, primals_120, primals_121, buf166, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 256, 1, 1), (256, 1, 256, 256))
        buf174 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf175 = reinterpret_tensor(buf174, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_24, silu_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf175, buf173, primals_123, primals_124, primals_125, primals_126, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 256, 1, 1), (256, 1, 256, 256))
        buf177 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf178 = reinterpret_tensor(buf177, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_25, y_20, y_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf178, buf176, primals_128, primals_129, primals_130, primals_131, buf172, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 256, 1, 1), (256, 1, 256, 256))
        buf180 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf181 = reinterpret_tensor(buf180, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_26, silu_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf181, buf179, primals_133, primals_134, primals_135, primals_136, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 256, 1, 1), (256, 1, 256, 256))
        buf183 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_27], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf182, primals_138, primals_139, primals_140, primals_141, buf183, 1024, grid=grid(1024), stream=stream0)
        buf184 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf183, buf178, buf112, buf184, 2048, grid=grid(2048), stream=stream0)
        del buf112
        del buf183
        # Topologically Sorted Source Nodes: [conv2d_28], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 512, 1, 1), (512, 1, 512, 512))
        buf186 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf187 = reinterpret_tensor(buf186, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_28, input_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf187, buf185, primals_143, primals_144, primals_145, primals_146, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_29], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, buf14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf189 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf190 = reinterpret_tensor(buf189, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_29, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf190, buf188, primals_148, primals_149, primals_150, primals_151, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_30], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 512, 1, 1), (512, 1, 512, 512))
        buf192 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf193 = reinterpret_tensor(buf192, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_30, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf193, buf191, primals_153, primals_154, primals_155, primals_156, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf190, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 512, 1, 1), (512, 1, 512, 512))
        buf195 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_31], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf194, primals_158, primals_159, primals_160, primals_161, buf195, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf193, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 512, 1, 1), (512, 1, 512, 512))
        buf197 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf198 = reinterpret_tensor(buf197, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_32, silu_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf198, buf196, primals_163, primals_164, primals_165, primals_166, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 512, 1, 1), (512, 1, 512, 512))
        buf200 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf201 = reinterpret_tensor(buf200, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_33, y_24, y_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf201, buf199, primals_168, primals_169, primals_170, primals_171, buf193, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 512, 1, 1), (512, 1, 512, 512))
        buf203 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf204 = reinterpret_tensor(buf203, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_34, silu_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf204, buf202, primals_173, primals_174, primals_175, primals_176, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_35], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 512, 1, 1), (512, 1, 512, 512))
        buf206 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf207 = reinterpret_tensor(buf206, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_35, y_26, y_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf207, buf205, primals_178, primals_179, primals_180, primals_181, buf201, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_36], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 512, 1, 1), (512, 1, 512, 512))
        buf209 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf210 = reinterpret_tensor(buf209, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_36, silu_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf210, buf208, primals_183, primals_184, primals_185, primals_186, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 512, 1, 1), (512, 1, 512, 512))
        buf212 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf213 = reinterpret_tensor(buf212, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_37, y_28, y_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf213, buf211, primals_188, primals_189, primals_190, primals_191, buf207, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 512, 1, 1), (512, 1, 512, 512))
        buf215 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf216 = reinterpret_tensor(buf215, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_38, silu_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf216, buf214, primals_193, primals_194, primals_195, primals_196, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 512, 1, 1), (512, 1, 512, 512))
        buf218 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf219 = reinterpret_tensor(buf218, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_39, y_30, y_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf219, buf217, primals_198, primals_199, primals_200, primals_201, buf213, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_40], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 512, 1, 1), (512, 1, 512, 512))
        buf221 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf222 = reinterpret_tensor(buf221, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_40, silu_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf222, buf220, primals_203, primals_204, primals_205, primals_206, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_41], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 512, 1, 1), (512, 1, 512, 512))
        buf224 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf225 = reinterpret_tensor(buf224, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_41, y_32, y_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf225, buf223, primals_208, primals_209, primals_210, primals_211, buf219, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_42], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 512, 1, 1), (512, 1, 512, 512))
        buf227 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf228 = reinterpret_tensor(buf227, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_42, silu_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf228, buf226, primals_213, primals_214, primals_215, primals_216, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_43], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 512, 1, 1), (512, 1, 512, 512))
        buf230 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf231 = reinterpret_tensor(buf230, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_43, y_34, y_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf231, buf229, primals_218, primals_219, primals_220, primals_221, buf225, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_44], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 512, 1, 1), (512, 1, 512, 512))
        buf233 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf234 = reinterpret_tensor(buf233, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_44, silu_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf234, buf232, primals_223, primals_224, primals_225, primals_226, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_45], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 512, 1, 1), (512, 1, 512, 512))
        buf236 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf237 = reinterpret_tensor(buf236, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_45, y_36, y_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf237, buf235, primals_228, primals_229, primals_230, primals_231, buf231, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_46], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 512, 1, 1), (512, 1, 512, 512))
        buf239 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf240 = reinterpret_tensor(buf239, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_46, silu_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf240, buf238, primals_233, primals_234, primals_235, primals_236, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_47], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 512, 1, 1), (512, 1, 512, 512))
        buf242 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf243 = reinterpret_tensor(buf242, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_47, y_38, y_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf243, buf241, primals_238, primals_239, primals_240, primals_241, buf237, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 512, 1, 1), (512, 1, 512, 512))
        buf245 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf246 = reinterpret_tensor(buf245, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_48, silu_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf246, buf244, primals_243, primals_244, primals_245, primals_246, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_49], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 512, 1, 1), (512, 1, 512, 512))
        buf248 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf249 = reinterpret_tensor(buf248, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_49, y_40, y_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf249, buf247, primals_248, primals_249, primals_250, primals_251, buf243, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_50], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 512, 1, 1), (512, 1, 512, 512))
        buf251 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf252 = reinterpret_tensor(buf251, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_50, silu_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf252, buf250, primals_253, primals_254, primals_255, primals_256, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 512, 1, 1), (512, 1, 512, 512))
        buf254 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf255 = reinterpret_tensor(buf254, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_51, y_42, y_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf255, buf253, primals_258, primals_259, primals_260, primals_261, buf249, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_52], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 512, 1, 1), (512, 1, 512, 512))
        buf257 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf258 = reinterpret_tensor(buf257, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_52, silu_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf258, buf256, primals_263, primals_264, primals_265, primals_266, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_53], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (4, 512, 1, 1), (512, 1, 512, 512))
        buf260 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf261 = reinterpret_tensor(buf260, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_53, y_44, y_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf261, buf259, primals_268, primals_269, primals_270, primals_271, buf255, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_54], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 512, 1, 1), (512, 1, 512, 512))
        buf263 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf264 = reinterpret_tensor(buf263, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_54, silu_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf264, buf262, primals_273, primals_274, primals_275, primals_276, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_55], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 512, 1, 1), (512, 1, 512, 512))
        buf266 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf267 = reinterpret_tensor(buf266, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_55, y_46, y_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf267, buf265, primals_278, primals_279, primals_280, primals_281, buf261, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 512, 1, 1), (512, 1, 512, 512))
        buf269 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf270 = reinterpret_tensor(buf269, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_56, silu_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf270, buf268, primals_283, primals_284, primals_285, primals_286, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 512, 1, 1), (512, 1, 512, 512))
        buf272 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf273 = reinterpret_tensor(buf272, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_57, y_48, y_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf273, buf271, primals_288, primals_289, primals_290, primals_291, buf267, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 512, 1, 1), (512, 1, 512, 512))
        buf275 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf276 = reinterpret_tensor(buf275, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_58, silu_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf276, buf274, primals_293, primals_294, primals_295, primals_296, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_59], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 512, 1, 1), (512, 1, 512, 512))
        buf278 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf279 = reinterpret_tensor(buf278, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_59, y_50, y_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf279, buf277, primals_298, primals_299, primals_300, primals_301, buf273, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_60], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 512, 1, 1), (512, 1, 512, 512))
        buf281 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf282 = reinterpret_tensor(buf281, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_60, silu_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf282, buf280, primals_303, primals_304, primals_305, primals_306, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_61], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 512, 1, 1), (512, 1, 512, 512))
        buf284 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf285 = reinterpret_tensor(buf284, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_61, y_52, y_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf285, buf283, primals_308, primals_309, primals_310, primals_311, buf279, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_62], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_312, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 512, 1, 1), (512, 1, 512, 512))
        buf287 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf288 = reinterpret_tensor(buf287, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_62, silu_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf288, buf286, primals_313, primals_314, primals_315, primals_316, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_63], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (4, 512, 1, 1), (512, 1, 512, 512))
        buf290 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf291 = reinterpret_tensor(buf290, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_63, y_54, y_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf291, buf289, primals_318, primals_319, primals_320, primals_321, buf285, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 512, 1, 1), (512, 1, 512, 512))
        buf293 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf294 = reinterpret_tensor(buf293, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_64, silu_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf294, buf292, primals_323, primals_324, primals_325, primals_326, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_65], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 512, 1, 1), (512, 1, 512, 512))
        buf296 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf297 = reinterpret_tensor(buf296, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_65, y_56, y_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf297, buf295, primals_328, primals_329, primals_330, primals_331, buf291, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_66], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 512, 1, 1), (512, 1, 512, 512))
        buf299 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf300 = reinterpret_tensor(buf299, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_66, silu_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf300, buf298, primals_333, primals_334, primals_335, primals_336, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_67], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 512, 1, 1), (512, 1, 512, 512))
        buf302 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf303 = reinterpret_tensor(buf302, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_67, y_58, y_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf303, buf301, primals_338, primals_339, primals_340, primals_341, buf297, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_68], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_342, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 512, 1, 1), (512, 1, 512, 512))
        buf305 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf306 = reinterpret_tensor(buf305, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_68, silu_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf306, buf304, primals_343, primals_344, primals_345, primals_346, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_69], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (4, 512, 1, 1), (512, 1, 512, 512))
        buf308 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf309 = reinterpret_tensor(buf308, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_69, y_60, y_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf309, buf307, primals_348, primals_349, primals_350, primals_351, buf303, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_70], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_352, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 512, 1, 1), (512, 1, 512, 512))
        buf311 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf312 = reinterpret_tensor(buf311, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_70, silu_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf312, buf310, primals_353, primals_354, primals_355, primals_356, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_71], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 512, 1, 1), (512, 1, 512, 512))
        buf314 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf315 = reinterpret_tensor(buf314, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_71, y_62, y_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf315, buf313, primals_358, primals_359, primals_360, primals_361, buf309, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_72], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, primals_362, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 512, 1, 1), (512, 1, 512, 512))
        buf317 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf318 = reinterpret_tensor(buf317, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_72, silu_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf318, buf316, primals_363, primals_364, primals_365, primals_366, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_73], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 512, 1, 1), (512, 1, 512, 512))
        buf320 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf321 = reinterpret_tensor(buf320, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_73, y_64, y_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf321, buf319, primals_368, primals_369, primals_370, primals_371, buf315, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_74], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 512, 1, 1), (512, 1, 512, 512))
        buf323 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf324 = reinterpret_tensor(buf323, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_74, silu_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf324, buf322, primals_373, primals_374, primals_375, primals_376, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_75], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf324, buf36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (4, 512, 1, 1), (512, 1, 512, 512))
        buf326 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf327 = reinterpret_tensor(buf326, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_75, y_66, y_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf327, buf325, primals_378, primals_379, primals_380, primals_381, buf321, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_76], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_382, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (4, 512, 1, 1), (512, 1, 512, 512))
        buf329 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf330 = reinterpret_tensor(buf329, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_76, silu_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf330, buf328, primals_383, primals_384, primals_385, primals_386, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_77], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 512, 1, 1), (512, 1, 512, 512))
        buf332 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf333 = reinterpret_tensor(buf332, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_77, y_68, y_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf333, buf331, primals_388, primals_389, primals_390, primals_391, buf327, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_78], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, primals_392, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 512, 1, 1), (512, 1, 512, 512))
        buf335 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf336 = reinterpret_tensor(buf335, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_78, silu_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf336, buf334, primals_393, primals_394, primals_395, primals_396, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_79], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (4, 512, 1, 1), (512, 1, 512, 512))
        buf338 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf339 = reinterpret_tensor(buf338, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_79, y_70, y_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf339, buf337, primals_398, primals_399, primals_400, primals_401, buf333, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_80], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, primals_402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 512, 1, 1), (512, 1, 512, 512))
        buf341 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf342 = reinterpret_tensor(buf341, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_80, silu_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf342, buf340, primals_403, primals_404, primals_405, primals_406, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_81], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (4, 512, 1, 1), (512, 1, 512, 512))
        buf344 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf345 = reinterpret_tensor(buf344, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_81, y_72, y_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf345, buf343, primals_408, primals_409, primals_410, primals_411, buf339, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_82], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_412, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (4, 512, 1, 1), (512, 1, 512, 512))
        buf347 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf348 = reinterpret_tensor(buf347, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_82, silu_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf348, buf346, primals_413, primals_414, primals_415, primals_416, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_83], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 512, 1, 1), (512, 1, 512, 512))
        buf350 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf351 = reinterpret_tensor(buf350, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_83, y_74, y_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf351, buf349, primals_418, primals_419, primals_420, primals_421, buf345, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_84], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (4, 512, 1, 1), (512, 1, 512, 512))
        buf353 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf354 = reinterpret_tensor(buf353, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_84, silu_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf354, buf352, primals_423, primals_424, primals_425, primals_426, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_85], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (4, 512, 1, 1), (512, 1, 512, 512))
        buf356 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf357 = reinterpret_tensor(buf356, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_85, y_76, y_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf357, buf355, primals_428, primals_429, primals_430, primals_431, buf351, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_86], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, primals_432, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 512, 1, 1), (512, 1, 512, 512))
        buf359 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf360 = reinterpret_tensor(buf359, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_86, silu_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf360, buf358, primals_433, primals_434, primals_435, primals_436, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_87], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (4, 512, 1, 1), (512, 1, 512, 512))
        buf362 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf363 = reinterpret_tensor(buf362, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_87, y_78, y_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf363, buf361, primals_438, primals_439, primals_440, primals_441, buf357, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_88], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, primals_442, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (4, 512, 1, 1), (512, 1, 512, 512))
        buf365 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf366 = reinterpret_tensor(buf365, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_88, silu_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf366, buf364, primals_443, primals_444, primals_445, primals_446, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_89], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (4, 512, 1, 1), (512, 1, 512, 512))
        buf368 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf369 = reinterpret_tensor(buf368, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf368  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_89, y_80, y_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf369, buf367, primals_448, primals_449, primals_450, primals_451, buf363, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_90], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(buf369, primals_452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (4, 512, 1, 1), (512, 1, 512, 512))
        buf371 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf372 = reinterpret_tensor(buf371, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf371  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_90, silu_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf372, buf370, primals_453, primals_454, primals_455, primals_456, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_91], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (4, 512, 1, 1), (512, 1, 512, 512))
        buf374 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf375 = reinterpret_tensor(buf374, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_91, y_82, y_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf375, buf373, primals_458, primals_459, primals_460, primals_461, buf369, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_92], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_462, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (4, 512, 1, 1), (512, 1, 512, 512))
        buf377 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf378 = reinterpret_tensor(buf377, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_92, silu_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf378, buf376, primals_463, primals_464, primals_465, primals_466, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_93], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (4, 512, 1, 1), (512, 1, 512, 512))
        buf380 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf381 = reinterpret_tensor(buf380, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_93, y_84, y_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf381, buf379, primals_468, primals_469, primals_470, primals_471, buf375, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_94], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (4, 512, 1, 1), (512, 1, 512, 512))
        buf383 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf384 = reinterpret_tensor(buf383, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_94, silu_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf384, buf382, primals_473, primals_474, primals_475, primals_476, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_95], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 512, 1, 1), (512, 1, 512, 512))
        buf386 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf387 = reinterpret_tensor(buf386, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_95, y_86, y_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf387, buf385, primals_478, primals_479, primals_480, primals_481, buf381, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_96], Original ATen: [aten.convolution]
        buf388 = extern_kernels.convolution(buf387, primals_482, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (4, 512, 1, 1), (512, 1, 512, 512))
        buf389 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf390 = reinterpret_tensor(buf389, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_96, silu_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf390, buf388, primals_483, primals_484, primals_485, primals_486, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_97], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf390, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (4, 512, 1, 1), (512, 1, 512, 512))
        buf392 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf393 = reinterpret_tensor(buf392, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_97, y_88, y_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf393, buf391, primals_488, primals_489, primals_490, primals_491, buf387, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_98], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, primals_492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (4, 512, 1, 1), (512, 1, 512, 512))
        buf395 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf396 = reinterpret_tensor(buf395, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_98, silu_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf396, buf394, primals_493, primals_494, primals_495, primals_496, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_99], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 512, 1, 1), (512, 1, 512, 512))
        buf398 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf399 = reinterpret_tensor(buf398, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_99, y_90, y_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf399, buf397, primals_498, primals_499, primals_500, primals_501, buf393, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_100], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf399, primals_502, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (4, 512, 1, 1), (512, 1, 512, 512))
        buf401 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf402 = reinterpret_tensor(buf401, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_100, silu_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf402, buf400, primals_503, primals_504, primals_505, primals_506, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_101], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (4, 512, 1, 1), (512, 1, 512, 512))
        buf404 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf405 = reinterpret_tensor(buf404, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_101, y_92, y_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf405, buf403, primals_508, primals_509, primals_510, primals_511, buf399, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_102], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf405, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (4, 512, 1, 1), (512, 1, 512, 512))
        buf407 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf408 = reinterpret_tensor(buf407, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_102, silu_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_11.run(buf408, buf406, primals_513, primals_514, primals_515, primals_516, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_103], Original ATen: [aten.convolution]
        buf409 = extern_kernels.convolution(buf408, buf50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf409, (4, 512, 1, 1), (512, 1, 512, 512))
        buf410 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_103], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf409, primals_518, primals_519, primals_520, primals_521, buf410, 2048, grid=grid(2048), stream=stream0)
        buf411 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf410, buf405, buf195, buf411, 4096, grid=grid(4096), stream=stream0)
        del buf195
        del buf410
        # Topologically Sorted Source Nodes: [conv2d_104], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, primals_522, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf413 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf414 = reinterpret_tensor(buf413, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_104, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf414, buf412, primals_523, primals_524, primals_525, primals_526, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_105], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf414, buf51, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf416 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf417 = reinterpret_tensor(buf416, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_105, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf417, buf415, primals_528, primals_529, primals_530, primals_531, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_106], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf417, primals_532, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf419 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf420 = reinterpret_tensor(buf419, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_106, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf420, buf418, primals_533, primals_534, primals_535, primals_536, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_107], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf417, primals_537, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf422 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_107], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf421, primals_538, primals_539, primals_540, primals_541, buf422, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_108], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf420, primals_542, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf423, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf424 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf425 = reinterpret_tensor(buf424, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_108, silu_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf425, buf423, primals_543, primals_544, primals_545, primals_546, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_109], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf427 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf428 = reinterpret_tensor(buf427, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_109, y_96, y_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf428, buf426, primals_548, primals_549, primals_550, primals_551, buf420, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_110], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf428, primals_552, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf430 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf431 = reinterpret_tensor(buf430, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf430  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_110, silu_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf431, buf429, primals_553, primals_554, primals_555, primals_556, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_111], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf433 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf434 = reinterpret_tensor(buf433, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_111, y_98, y_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf434, buf432, primals_558, primals_559, primals_560, primals_561, buf428, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_112], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf434, primals_562, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf436 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf437 = reinterpret_tensor(buf436, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_112, silu_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf437, buf435, primals_563, primals_564, primals_565, primals_566, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_113], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf439 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf440 = reinterpret_tensor(buf439, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_113, y_100, y_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf440, buf438, primals_568, primals_569, primals_570, primals_571, buf434, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_114], Original ATen: [aten.convolution]
        buf441 = extern_kernels.convolution(buf440, primals_572, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf441, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf442 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf443 = reinterpret_tensor(buf442, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_114, silu_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf443, buf441, primals_573, primals_574, primals_575, primals_576, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_115], Original ATen: [aten.convolution]
        buf444 = extern_kernels.convolution(buf443, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf444, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf445 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf446 = reinterpret_tensor(buf445, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_115, y_102, y_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf446, buf444, primals_578, primals_579, primals_580, primals_581, buf440, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_116], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, primals_582, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf448 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf449 = reinterpret_tensor(buf448, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_116, silu_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf449, buf447, primals_583, primals_584, primals_585, primals_586, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_117], Original ATen: [aten.convolution]
        buf450 = extern_kernels.convolution(buf449, buf56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf451 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf452 = reinterpret_tensor(buf451, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_117, y_104, y_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf452, buf450, primals_588, primals_589, primals_590, primals_591, buf446, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_118], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_592, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf454 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf455 = reinterpret_tensor(buf454, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_118, silu_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf455, buf453, primals_593, primals_594, primals_595, primals_596, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_119], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf457 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf458 = reinterpret_tensor(buf457, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_119, y_106, y_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf458, buf456, primals_598, primals_599, primals_600, primals_601, buf452, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_120], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf458, primals_602, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf460 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf461 = reinterpret_tensor(buf460, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_120, silu_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf461, buf459, primals_603, primals_604, primals_605, primals_606, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_121], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf463 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf464 = reinterpret_tensor(buf463, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_121, y_108, y_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf464, buf462, primals_608, primals_609, primals_610, primals_611, buf458, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_122], Original ATen: [aten.convolution]
        buf465 = extern_kernels.convolution(buf464, primals_612, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf466 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf467 = reinterpret_tensor(buf466, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_122, silu_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf467, buf465, primals_613, primals_614, primals_615, primals_616, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_123], Original ATen: [aten.convolution]
        buf468 = extern_kernels.convolution(buf467, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf468, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf469 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf470 = reinterpret_tensor(buf469, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_123, y_110, y_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf470, buf468, primals_618, primals_619, primals_620, primals_621, buf464, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_124], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, primals_622, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf472 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf473 = reinterpret_tensor(buf472, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_124, silu_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf473, buf471, primals_623, primals_624, primals_625, primals_626, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_125], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, buf60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf475 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf476 = reinterpret_tensor(buf475, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf475  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_125, y_112, y_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf476, buf474, primals_628, primals_629, primals_630, primals_631, buf470, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_126], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, primals_632, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf478 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf479 = reinterpret_tensor(buf478, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_126, silu_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf479, buf477, primals_633, primals_634, primals_635, primals_636, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_127], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf479, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf481 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf482 = reinterpret_tensor(buf481, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf481  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_127, y_114, y_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf482, buf480, primals_638, primals_639, primals_640, primals_641, buf476, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_128], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_642, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf484 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf485 = reinterpret_tensor(buf484, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_128, silu_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf485, buf483, primals_643, primals_644, primals_645, primals_646, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_129], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, buf62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf487 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf488 = reinterpret_tensor(buf487, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_129, y_116, y_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf488, buf486, primals_648, primals_649, primals_650, primals_651, buf482, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_130], Original ATen: [aten.convolution]
        buf489 = extern_kernels.convolution(buf488, primals_652, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf489, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf490 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf491 = reinterpret_tensor(buf490, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_130, silu_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf491, buf489, primals_653, primals_654, primals_655, primals_656, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_131], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf491, buf63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf493 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf494 = reinterpret_tensor(buf493, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_131, y_118, y_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf494, buf492, primals_658, primals_659, primals_660, primals_661, buf488, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_132], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, primals_662, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf496 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf497 = reinterpret_tensor(buf496, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_132, silu_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf497, buf495, primals_663, primals_664, primals_665, primals_666, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_133], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf499 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf500 = reinterpret_tensor(buf499, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_133, y_120, y_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf500, buf498, primals_668, primals_669, primals_670, primals_671, buf494, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_134], Original ATen: [aten.convolution]
        buf501 = extern_kernels.convolution(buf500, primals_672, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf502 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf503 = reinterpret_tensor(buf502, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_134, silu_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf503, buf501, primals_673, primals_674, primals_675, primals_676, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_135], Original ATen: [aten.convolution]
        buf504 = extern_kernels.convolution(buf503, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf504, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf505 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf506 = reinterpret_tensor(buf505, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_135, y_122, y_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf506, buf504, primals_678, primals_679, primals_680, primals_681, buf500, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_136], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf506, primals_682, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf508 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf509 = reinterpret_tensor(buf508, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_136, silu_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf509, buf507, primals_683, primals_684, primals_685, primals_686, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_137], Original ATen: [aten.convolution]
        buf510 = extern_kernels.convolution(buf509, buf66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf510, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf511 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf512 = reinterpret_tensor(buf511, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_137, y_124, y_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf512, buf510, primals_688, primals_689, primals_690, primals_691, buf506, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_138], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf512, primals_692, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf514 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf515 = reinterpret_tensor(buf514, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_138, silu_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf515, buf513, primals_693, primals_694, primals_695, primals_696, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_139], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, buf67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf517 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf518 = reinterpret_tensor(buf517, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_139, y_126, y_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf518, buf516, primals_698, primals_699, primals_700, primals_701, buf512, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_140], Original ATen: [aten.convolution]
        buf519 = extern_kernels.convolution(buf518, primals_702, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf520 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf521 = reinterpret_tensor(buf520, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_140, silu_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf521, buf519, primals_703, primals_704, primals_705, primals_706, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_141], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf523 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf524 = reinterpret_tensor(buf523, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_141, y_128, y_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf524, buf522, primals_708, primals_709, primals_710, primals_711, buf518, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_142], Original ATen: [aten.convolution]
        buf525 = extern_kernels.convolution(buf524, primals_712, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf525, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf526 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf527 = reinterpret_tensor(buf526, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_142, silu_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf527, buf525, primals_713, primals_714, primals_715, primals_716, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_143], Original ATen: [aten.convolution]
        buf528 = extern_kernels.convolution(buf527, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf528, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf529 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf530 = reinterpret_tensor(buf529, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_143, y_130, y_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf530, buf528, primals_718, primals_719, primals_720, primals_721, buf524, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_144], Original ATen: [aten.convolution]
        buf531 = extern_kernels.convolution(buf530, primals_722, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf531, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf532 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf533 = reinterpret_tensor(buf532, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_144, silu_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf533, buf531, primals_723, primals_724, primals_725, primals_726, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_145], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, buf70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf535 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf536 = reinterpret_tensor(buf535, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf535  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_145, y_132, y_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf536, buf534, primals_728, primals_729, primals_730, primals_731, buf530, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_146], Original ATen: [aten.convolution]
        buf537 = extern_kernels.convolution(buf536, primals_732, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf537, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf538 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf539 = reinterpret_tensor(buf538, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf538  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_146, silu_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf539, buf537, primals_733, primals_734, primals_735, primals_736, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_147], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf541 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf542 = reinterpret_tensor(buf541, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_147, y_134, y_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf542, buf540, primals_738, primals_739, primals_740, primals_741, buf536, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_148], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf542, primals_742, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf544 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf545 = reinterpret_tensor(buf544, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf544  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_148, silu_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf545, buf543, primals_743, primals_744, primals_745, primals_746, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_149], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf545, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf547 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf548 = reinterpret_tensor(buf547, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_149, y_136, y_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf548, buf546, primals_748, primals_749, primals_750, primals_751, buf542, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_150], Original ATen: [aten.convolution]
        buf549 = extern_kernels.convolution(buf548, primals_752, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf549, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf550 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf551 = reinterpret_tensor(buf550, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf550  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_150, silu_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf551, buf549, primals_753, primals_754, primals_755, primals_756, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_151], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf553 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf554 = reinterpret_tensor(buf553, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf553  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_151, y_138, y_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf554, buf552, primals_758, primals_759, primals_760, primals_761, buf548, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_152], Original ATen: [aten.convolution]
        buf555 = extern_kernels.convolution(buf554, primals_762, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf555, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf556 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf557 = reinterpret_tensor(buf556, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_152, silu_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf557, buf555, primals_763, primals_764, primals_765, primals_766, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_153], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, buf74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf559 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf560 = reinterpret_tensor(buf559, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_153, y_140, y_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf560, buf558, primals_768, primals_769, primals_770, primals_771, buf554, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_154], Original ATen: [aten.convolution]
        buf561 = extern_kernels.convolution(buf560, primals_772, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf561, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf562 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf563 = reinterpret_tensor(buf562, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_154, silu_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf563, buf561, primals_773, primals_774, primals_775, primals_776, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_155], Original ATen: [aten.convolution]
        buf564 = extern_kernels.convolution(buf563, buf75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf564, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf565 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf566 = reinterpret_tensor(buf565, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf565  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_155, y_142, y_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf566, buf564, primals_778, primals_779, primals_780, primals_781, buf560, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_156], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(buf566, primals_782, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf568 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf569 = reinterpret_tensor(buf568, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf568  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_156, silu_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf569, buf567, primals_783, primals_784, primals_785, primals_786, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_157], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf571 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf572 = reinterpret_tensor(buf571, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_157, y_144, y_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf572, buf570, primals_788, primals_789, primals_790, primals_791, buf566, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_158], Original ATen: [aten.convolution]
        buf573 = extern_kernels.convolution(buf572, primals_792, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf573, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf574 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf575 = reinterpret_tensor(buf574, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_158, silu_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf575, buf573, primals_793, primals_794, primals_795, primals_796, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_159], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf577 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf578 = reinterpret_tensor(buf577, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_159, y_146, y_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf578, buf576, primals_798, primals_799, primals_800, primals_801, buf572, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_160], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, primals_802, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf580 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf581 = reinterpret_tensor(buf580, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_160, silu_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf581, buf579, primals_803, primals_804, primals_805, primals_806, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_161], Original ATen: [aten.convolution]
        buf582 = extern_kernels.convolution(buf581, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf582, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf583 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf584 = reinterpret_tensor(buf583, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_161, y_148, y_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf584, buf582, primals_808, primals_809, primals_810, primals_811, buf578, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_162], Original ATen: [aten.convolution]
        buf585 = extern_kernels.convolution(buf584, primals_812, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf585, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf586 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf587 = reinterpret_tensor(buf586, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_162, silu_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf587, buf585, primals_813, primals_814, primals_815, primals_816, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_163], Original ATen: [aten.convolution]
        buf588 = extern_kernels.convolution(buf587, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf588, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf589 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf590 = reinterpret_tensor(buf589, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf589  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_163, y_150, y_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf590, buf588, primals_818, primals_819, primals_820, primals_821, buf584, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_164], Original ATen: [aten.convolution]
        buf591 = extern_kernels.convolution(buf590, primals_822, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf591, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf592 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf593 = reinterpret_tensor(buf592, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_164, silu_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf593, buf591, primals_823, primals_824, primals_825, primals_826, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_165], Original ATen: [aten.convolution]
        buf594 = extern_kernels.convolution(buf593, buf80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf594, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf595 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf596 = reinterpret_tensor(buf595, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_165, y_152, y_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf596, buf594, primals_828, primals_829, primals_830, primals_831, buf590, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_166], Original ATen: [aten.convolution]
        buf597 = extern_kernels.convolution(buf596, primals_832, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf597, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf598 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf599 = reinterpret_tensor(buf598, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_166, silu_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf599, buf597, primals_833, primals_834, primals_835, primals_836, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_167], Original ATen: [aten.convolution]
        buf600 = extern_kernels.convolution(buf599, buf81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf601 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf602 = reinterpret_tensor(buf601, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf601  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_167, y_154, y_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf602, buf600, primals_838, primals_839, primals_840, primals_841, buf596, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_168], Original ATen: [aten.convolution]
        buf603 = extern_kernels.convolution(buf602, primals_842, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf603, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf604 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf605 = reinterpret_tensor(buf604, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf604  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_168, silu_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf605, buf603, primals_843, primals_844, primals_845, primals_846, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_169], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, buf82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf607 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf608 = reinterpret_tensor(buf607, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf607  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_169, y_156, y_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf608, buf606, primals_848, primals_849, primals_850, primals_851, buf602, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_170], Original ATen: [aten.convolution]
        buf609 = extern_kernels.convolution(buf608, primals_852, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf609, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf610 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf611 = reinterpret_tensor(buf610, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf610  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_170, silu_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf611, buf609, primals_853, primals_854, primals_855, primals_856, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_171], Original ATen: [aten.convolution]
        buf612 = extern_kernels.convolution(buf611, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf612, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf613 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf614 = reinterpret_tensor(buf613, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_171, y_158, y_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf614, buf612, primals_858, primals_859, primals_860, primals_861, buf608, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_172], Original ATen: [aten.convolution]
        buf615 = extern_kernels.convolution(buf614, primals_862, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf615, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf616 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf617 = reinterpret_tensor(buf616, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf616  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_172, silu_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf617, buf615, primals_863, primals_864, primals_865, primals_866, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_173], Original ATen: [aten.convolution]
        buf618 = extern_kernels.convolution(buf617, buf84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf619 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf620 = reinterpret_tensor(buf619, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf619  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_173, y_160, y_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf620, buf618, primals_868, primals_869, primals_870, primals_871, buf614, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_174], Original ATen: [aten.convolution]
        buf621 = extern_kernels.convolution(buf620, primals_872, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf621, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf622 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf623 = reinterpret_tensor(buf622, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf622  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_174, silu_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf623, buf621, primals_873, primals_874, primals_875, primals_876, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_175], Original ATen: [aten.convolution]
        buf624 = extern_kernels.convolution(buf623, buf85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf624, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf625 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf626 = reinterpret_tensor(buf625, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf625  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_175, y_162, y_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf626, buf624, primals_878, primals_879, primals_880, primals_881, buf620, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_176], Original ATen: [aten.convolution]
        buf627 = extern_kernels.convolution(buf626, primals_882, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf627, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf628 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf629 = reinterpret_tensor(buf628, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf628  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_176, silu_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf629, buf627, primals_883, primals_884, primals_885, primals_886, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_177], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, buf86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf631 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf632 = reinterpret_tensor(buf631, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf631  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_177, y_164, y_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_22.run(buf632, buf630, primals_888, primals_889, primals_890, primals_891, buf626, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_178], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf632, primals_892, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf634 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf635 = reinterpret_tensor(buf634, (4, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf634  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_178, silu_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf635, buf633, primals_893, primals_894, primals_895, primals_896, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_179], Original ATen: [aten.convolution]
        buf636 = extern_kernels.convolution(buf635, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf636, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
        buf637 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_179], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf636, primals_898, primals_899, primals_900, primals_901, buf637, 4096, grid=grid(4096), stream=stream0)
        buf638 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf637, buf632, buf422, buf638, 8192, grid=grid(8192), stream=stream0)
        del buf422
        del buf637
        # Topologically Sorted Source Nodes: [conv2d_180], Original ATen: [aten.convolution]
        buf639 = extern_kernels.convolution(buf638, primals_902, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf639, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf640 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf641 = reinterpret_tensor(buf640, (4, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf640  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_180, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf641, buf639, primals_903, primals_904, primals_905, primals_906, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_181], Original ATen: [aten.convolution]
        buf642 = extern_kernels.convolution(buf641, buf88, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf642, (4, 4096, 1, 1), (4096, 1, 4096, 4096))
        buf643 = empty_strided_cuda((4, 4096, 1, 1), (4096, 1, 16384, 16384), torch.float32)
        buf644 = reinterpret_tensor(buf643, (4, 4096, 1, 1), (4096, 1, 4096, 4096), 0); del buf643  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_181, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_24.run(buf644, buf642, primals_908, primals_909, primals_910, primals_911, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_182], Original ATen: [aten.convolution]
        buf645 = extern_kernels.convolution(buf644, primals_912, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf645, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf646 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf647 = reinterpret_tensor(buf646, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf646  # reuse
        buf648 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 2048, 2048), torch.int8)
        # Topologically Sorted Source Nodes: [batch_norm_182, x_11, max_pool2d], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_silu_25.run(buf647, buf645, primals_913, primals_914, primals_915, primals_916, buf648, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [max_pool2d_1], Original ATen: [aten.max_pool2d_with_indices]
        buf649 = torch.ops.aten.max_pool2d_with_indices.default(buf647, [9, 9], [1, 1], [4, 4])
        buf650 = buf649[0]
        buf651 = buf649[1]
        del buf649
        # Topologically Sorted Source Nodes: [max_pool2d_2], Original ATen: [aten.max_pool2d_with_indices]
        buf652 = torch.ops.aten.max_pool2d_with_indices.default(buf647, [13, 13], [1, 1], [6, 6])
        buf653 = buf652[0]
        buf654 = buf652[1]
        del buf652
        buf655 = empty_strided_cuda((4, 8192, 1, 1), (8192, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_26.run(buf647, buf650, buf653, buf655, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_183], Original ATen: [aten.convolution]
        buf656 = extern_kernels.convolution(buf655, primals_917, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf656, (4, 4096, 1, 1), (4096, 1, 4096, 4096))
        buf657 = empty_strided_cuda((4, 4096, 1, 1), (4096, 1, 16384, 16384), torch.float32)
        buf658 = reinterpret_tensor(buf657, (4, 4096, 1, 1), (4096, 1, 4096, 4096), 0); del buf657  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_183, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_24.run(buf658, buf656, primals_918, primals_919, primals_920, primals_921, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_184], Original ATen: [aten.convolution]
        buf659 = extern_kernels.convolution(buf658, primals_922, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf659, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf660 = reinterpret_tensor(buf653, (4, 2048, 1, 1), (2048, 1, 8192, 8192), 0); del buf653  # reuse
        buf661 = reinterpret_tensor(buf660, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf660  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_184, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf661, buf659, primals_923, primals_924, primals_925, primals_926, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_185], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(buf658, primals_927, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf662, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf663 = reinterpret_tensor(buf650, (4, 2048, 1, 1), (2048, 1, 8192, 8192), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_185], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf662, primals_928, primals_929, primals_930, primals_931, buf663, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_186], Original ATen: [aten.convolution]
        buf664 = extern_kernels.convolution(buf661, primals_932, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf664, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf665 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf666 = reinterpret_tensor(buf665, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf665  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_186, silu_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf666, buf664, primals_933, primals_934, primals_935, primals_936, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_187], Original ATen: [aten.convolution]
        buf667 = extern_kernels.convolution(buf666, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf667, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf668 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf669 = reinterpret_tensor(buf668, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf668  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_187, y_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf669, buf667, primals_938, primals_939, primals_940, primals_941, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_188], Original ATen: [aten.convolution]
        buf670 = extern_kernels.convolution(buf669, primals_942, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf670, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf671 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf672 = reinterpret_tensor(buf671, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf671  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_188, silu_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf672, buf670, primals_943, primals_944, primals_945, primals_946, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_189], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf672, buf90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf673, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf674 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf675 = reinterpret_tensor(buf674, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf674  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_189, y_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf675, buf673, primals_948, primals_949, primals_950, primals_951, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_190], Original ATen: [aten.convolution]
        buf676 = extern_kernels.convolution(buf675, primals_952, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf677 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf678 = reinterpret_tensor(buf677, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf677  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_190, silu_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf678, buf676, primals_953, primals_954, primals_955, primals_956, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_191], Original ATen: [aten.convolution]
        buf679 = extern_kernels.convolution(buf678, buf91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf679, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf680 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf681 = reinterpret_tensor(buf680, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf680  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_191, y_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf681, buf679, primals_958, primals_959, primals_960, primals_961, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_192], Original ATen: [aten.convolution]
        buf682 = extern_kernels.convolution(buf681, primals_962, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf682, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf683 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf684 = reinterpret_tensor(buf683, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf683  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_192, silu_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf684, buf682, primals_963, primals_964, primals_965, primals_966, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_193], Original ATen: [aten.convolution]
        buf685 = extern_kernels.convolution(buf684, buf92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf685, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf686 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf687 = reinterpret_tensor(buf686, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf686  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_193, y_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf687, buf685, primals_968, primals_969, primals_970, primals_971, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_194], Original ATen: [aten.convolution]
        buf688 = extern_kernels.convolution(buf687, primals_972, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf688, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf689 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf690 = reinterpret_tensor(buf689, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf689  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_194, silu_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf690, buf688, primals_973, primals_974, primals_975, primals_976, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_195], Original ATen: [aten.convolution]
        buf691 = extern_kernels.convolution(buf690, buf93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf691, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf692 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf693 = reinterpret_tensor(buf692, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf692  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_195, y_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf693, buf691, primals_978, primals_979, primals_980, primals_981, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_196], Original ATen: [aten.convolution]
        buf694 = extern_kernels.convolution(buf693, primals_982, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf694, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf695 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf696 = reinterpret_tensor(buf695, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf695  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_196, silu_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf696, buf694, primals_983, primals_984, primals_985, primals_986, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_197], Original ATen: [aten.convolution]
        buf697 = extern_kernels.convolution(buf696, buf94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf697, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf698 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf699 = reinterpret_tensor(buf698, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf698  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_197, y_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf699, buf697, primals_988, primals_989, primals_990, primals_991, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_198], Original ATen: [aten.convolution]
        buf700 = extern_kernels.convolution(buf699, primals_992, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf700, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf701 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf702 = reinterpret_tensor(buf701, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf701  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_198, silu_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf702, buf700, primals_993, primals_994, primals_995, primals_996, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_199], Original ATen: [aten.convolution]
        buf703 = extern_kernels.convolution(buf702, buf95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf703, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf704 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf705 = reinterpret_tensor(buf704, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf704  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_199, y_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf705, buf703, primals_998, primals_999, primals_1000, primals_1001, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_200], Original ATen: [aten.convolution]
        buf706 = extern_kernels.convolution(buf705, primals_1002, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf706, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf707 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf708 = reinterpret_tensor(buf707, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf707  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_200, silu_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf708, buf706, primals_1003, primals_1004, primals_1005, primals_1006, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_201], Original ATen: [aten.convolution]
        buf709 = extern_kernels.convolution(buf708, buf96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf709, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf710 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf711 = reinterpret_tensor(buf710, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf710  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_201, y_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf711, buf709, primals_1008, primals_1009, primals_1010, primals_1011, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_202], Original ATen: [aten.convolution]
        buf712 = extern_kernels.convolution(buf711, primals_1012, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf712, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf713 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf714 = reinterpret_tensor(buf713, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf713  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_202, silu_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf714, buf712, primals_1013, primals_1014, primals_1015, primals_1016, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_203], Original ATen: [aten.convolution]
        buf715 = extern_kernels.convolution(buf714, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf715, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf716 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf717 = reinterpret_tensor(buf716, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf716  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_203, y_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf717, buf715, primals_1018, primals_1019, primals_1020, primals_1021, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_204], Original ATen: [aten.convolution]
        buf718 = extern_kernels.convolution(buf717, primals_1022, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf718, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf719 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf720 = reinterpret_tensor(buf719, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf719  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_204, silu_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf720, buf718, primals_1023, primals_1024, primals_1025, primals_1026, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_205], Original ATen: [aten.convolution]
        buf721 = extern_kernels.convolution(buf720, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf721, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf722 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf723 = reinterpret_tensor(buf722, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf722  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_205, y_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf723, buf721, primals_1028, primals_1029, primals_1030, primals_1031, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_206], Original ATen: [aten.convolution]
        buf724 = extern_kernels.convolution(buf723, primals_1032, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf724, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf725 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf726 = reinterpret_tensor(buf725, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf725  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_206, silu_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf726, buf724, primals_1033, primals_1034, primals_1035, primals_1036, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_207], Original ATen: [aten.convolution]
        buf727 = extern_kernels.convolution(buf726, buf99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf727, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf728 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf729 = reinterpret_tensor(buf728, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf728  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_207, y_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf729, buf727, primals_1038, primals_1039, primals_1040, primals_1041, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_208], Original ATen: [aten.convolution]
        buf730 = extern_kernels.convolution(buf729, primals_1042, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf730, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf731 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf732 = reinterpret_tensor(buf731, (4, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf731  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_208, silu_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf732, buf730, primals_1043, primals_1044, primals_1045, primals_1046, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_209], Original ATen: [aten.convolution]
        buf733 = extern_kernels.convolution(buf732, buf100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf733, (4, 2048, 1, 1), (2048, 1, 2048, 2048))
        buf734 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_209], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf733, primals_1048, primals_1049, primals_1050, primals_1051, buf734, 8192, grid=grid(8192), stream=stream0)
        buf735 = empty_strided_cuda((4, 4096, 1, 1), (4096, 1, 4096, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf734, buf663, buf735, 16384, grid=grid(16384), stream=stream0)
        del buf663
        del buf734
        # Topologically Sorted Source Nodes: [conv2d_210], Original ATen: [aten.convolution]
        buf736 = extern_kernels.convolution(buf735, primals_1052, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf736, (4, 4096, 1, 1), (4096, 1, 4096, 4096))
        buf737 = empty_strided_cuda((4, 4096, 1, 1), (4096, 1, 16384, 16384), torch.float32)
        buf738 = reinterpret_tensor(buf737, (4, 4096, 1, 1), (4096, 1, 1, 1), 0); del buf737  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_210, input_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_24.run(buf738, buf736, primals_1053, primals_1054, primals_1055, primals_1056, 16384, grid=grid(16384), stream=stream0)
    return (buf414, buf641, buf738, buf0, primals_3, primals_4, primals_5, primals_6, buf1, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, buf2, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, buf3, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, buf4, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, buf5, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, buf6, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, buf7, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, buf8, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, buf9, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, buf10, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, buf11, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, buf12, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, buf13, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, buf14, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, buf15, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, buf16, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, buf17, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, buf18, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, buf19, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, buf20, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, buf21, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, buf22, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, buf23, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, buf24, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, buf25, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, buf26, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, buf27, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, buf28, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, buf29, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, buf30, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, buf31, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, buf32, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, buf33, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, buf34, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, buf35, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, buf36, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, buf37, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, buf38, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, buf39, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, buf40, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, buf41, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, buf42, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, buf43, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, buf44, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, buf45, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, buf46, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, buf47, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, buf48, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, buf49, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, buf50, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, buf51, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, buf52, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, buf53, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, buf54, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, buf55, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, buf56, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, buf57, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, buf58, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, buf59, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, buf60, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, buf61, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, buf62, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, buf63, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, buf64, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, buf65, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, buf66, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, buf67, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, buf68, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, buf69, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, buf70, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, buf71, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, buf72, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, buf73, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, buf74, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, buf75, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, buf76, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, buf77, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, buf78, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, buf79, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, buf80, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, buf81, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, buf82, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, buf83, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, buf84, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, buf85, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, buf86, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, buf87, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, buf88, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, buf89, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, buf90, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, buf91, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, buf92, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, buf93, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, buf94, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, buf95, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, buf96, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, buf97, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, buf98, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, buf99, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, buf100, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, buf101, buf102, buf104, buf105, buf107, buf108, buf110, buf111, buf113, buf115, buf116, buf118, buf119, buf121, buf122, buf124, buf125, buf127, buf128, buf130, buf131, buf133, buf134, buf136, buf137, buf139, buf140, buf142, buf143, buf145, buf146, buf148, buf149, buf151, buf152, buf154, buf155, buf157, buf158, buf160, buf161, buf163, buf164, buf166, buf167, buf169, buf170, buf172, buf173, buf175, buf176, buf178, buf179, buf181, buf182, buf184, buf185, buf187, buf188, buf190, buf191, buf193, buf194, buf196, buf198, buf199, buf201, buf202, buf204, buf205, buf207, buf208, buf210, buf211, buf213, buf214, buf216, buf217, buf219, buf220, buf222, buf223, buf225, buf226, buf228, buf229, buf231, buf232, buf234, buf235, buf237, buf238, buf240, buf241, buf243, buf244, buf246, buf247, buf249, buf250, buf252, buf253, buf255, buf256, buf258, buf259, buf261, buf262, buf264, buf265, buf267, buf268, buf270, buf271, buf273, buf274, buf276, buf277, buf279, buf280, buf282, buf283, buf285, buf286, buf288, buf289, buf291, buf292, buf294, buf295, buf297, buf298, buf300, buf301, buf303, buf304, buf306, buf307, buf309, buf310, buf312, buf313, buf315, buf316, buf318, buf319, buf321, buf322, buf324, buf325, buf327, buf328, buf330, buf331, buf333, buf334, buf336, buf337, buf339, buf340, buf342, buf343, buf345, buf346, buf348, buf349, buf351, buf352, buf354, buf355, buf357, buf358, buf360, buf361, buf363, buf364, buf366, buf367, buf369, buf370, buf372, buf373, buf375, buf376, buf378, buf379, buf381, buf382, buf384, buf385, buf387, buf388, buf390, buf391, buf393, buf394, buf396, buf397, buf399, buf400, buf402, buf403, buf405, buf406, buf408, buf409, buf411, buf412, buf414, buf415, buf417, buf418, buf420, buf421, buf423, buf425, buf426, buf428, buf429, buf431, buf432, buf434, buf435, buf437, buf438, buf440, buf441, buf443, buf444, buf446, buf447, buf449, buf450, buf452, buf453, buf455, buf456, buf458, buf459, buf461, buf462, buf464, buf465, buf467, buf468, buf470, buf471, buf473, buf474, buf476, buf477, buf479, buf480, buf482, buf483, buf485, buf486, buf488, buf489, buf491, buf492, buf494, buf495, buf497, buf498, buf500, buf501, buf503, buf504, buf506, buf507, buf509, buf510, buf512, buf513, buf515, buf516, buf518, buf519, buf521, buf522, buf524, buf525, buf527, buf528, buf530, buf531, buf533, buf534, buf536, buf537, buf539, buf540, buf542, buf543, buf545, buf546, buf548, buf549, buf551, buf552, buf554, buf555, buf557, buf558, buf560, buf561, buf563, buf564, buf566, buf567, buf569, buf570, buf572, buf573, buf575, buf576, buf578, buf579, buf581, buf582, buf584, buf585, buf587, buf588, buf590, buf591, buf593, buf594, buf596, buf597, buf599, buf600, buf602, buf603, buf605, buf606, buf608, buf609, buf611, buf612, buf614, buf615, buf617, buf618, buf620, buf621, buf623, buf624, buf626, buf627, buf629, buf630, buf632, buf633, buf635, buf636, buf638, buf639, buf641, buf642, buf644, buf645, buf647, buf648, buf651, buf654, buf655, buf656, buf658, buf659, buf661, buf662, buf664, buf666, buf667, buf669, buf670, buf672, buf673, buf675, buf676, buf678, buf679, buf681, buf682, buf684, buf685, buf687, buf688, buf690, buf691, buf693, buf694, buf696, buf697, buf699, buf700, buf702, buf703, buf705, buf706, buf708, buf709, buf711, buf712, buf714, buf715, buf717, buf718, buf720, buf721, buf723, buf724, buf726, buf727, buf729, buf730, buf732, buf733, buf735, buf736, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((256, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
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
    primals_357 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((2048, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((4096, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((4096, 8192, 1, 1), (8192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_924 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_927 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_930 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_933 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_938 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_939 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_940 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_941 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_942 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_943 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_944 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_945 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_946 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_947 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_948 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_949 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_950 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_951 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_952 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_953 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_954 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_955 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_956 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_957 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_958 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_959 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_960 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_961 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_962 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_963 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_964 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_965 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_966 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_967 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_968 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_969 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_970 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_971 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_972 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_973 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_974 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_975 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_976 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_977 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_978 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_979 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_980 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_981 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_982 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_983 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_984 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_985 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_986 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_987 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_988 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_989 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_990 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_991 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_992 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_993 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_994 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_995 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_996 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_997 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_998 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_999 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1000 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1001 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1002 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1003 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1004 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1005 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1006 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1007 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1008 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1009 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1010 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1011 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1012 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1013 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1014 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1015 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1016 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1017 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1018 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1019 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1020 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1021 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1022 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1023 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1024 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1025 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1026 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1027 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1028 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1029 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1030 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1031 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1032 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1033 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1034 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1035 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1036 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1037 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1038 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1039 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1040 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1041 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1042 = rand_strided((2048, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1043 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1044 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1045 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1046 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1047 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1048 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1049 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1050 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1051 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1052 = rand_strided((4096, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1053 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1054 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1055 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1056 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
