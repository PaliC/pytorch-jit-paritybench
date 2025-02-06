# AOT ID: ['8_forward']
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


# kernel path: inductor_cache/7f/c7fcjjxl72i5ydjufnmgp5pturl3b7sw5iwe7hjgp76bwr2elyac.py
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
    size_hints={'y': 128, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/w4/cw47rb5cf3aclnwupcad3in6l54aitsuyuaoxwmc5q3mzk7jgnuc.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2240
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 35)
    y1 = yindex // 35
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 35*x2 + 315*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/c6/cc64ibizyhcr4t4fywr4my3rl6v5jkz3tkwrbd4tbau5zujv6ni5.py
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
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 131)
    y1 = yindex // 131
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 131*x2 + 1179*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ua/cuakqpmezogzl6zo4onu5l2vaftk5xta42mnnf5ldo5pu6rf6b32.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%primals_1, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_5 = async_compile.triton('triton_poi_fused_avg_pool2d_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = ((yindex // 32) % 32)
    y0 = (yindex % 32)
    x3 = xindex
    y4 = yindex // 32
    y2 = yindex // 1024
    y5 = (yindex % 1024)
    tmp0 = (-1) + 2*y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*y0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-195) + x3 + 6*y0 + 384*y4), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-192) + x3 + 6*y0 + 384*y4), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*y0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-189) + x3 + 6*y0 + 384*y4), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-3) + x3 + 6*y0 + 384*y4), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x3 + 6*y0 + 384*y4), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (3 + x3 + 6*y0 + 384*y4), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*y1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (189 + x3 + 6*y0 + 384*y4), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (192 + x3 + 6*y0 + 384*y4), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (195 + x3 + 6*y0 + 384*y4), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + ((65) * ((65) <= (2 + 2*y0)) + (2 + 2*y0) * ((2 + 2*y0) < (65)))*((65) * ((65) <= (2 + 2*y1)) + (2 + 2*y1) * ((2 + 2*y1) < (65))) + ((-2)*y0*((65) * ((65) <= (2 + 2*y1)) + (2 + 2*y1) * ((2 + 2*y1) < (65)))) + ((-2)*y1*((65) * ((65) <= (2 + 2*y0)) + (2 + 2*y0) * ((2 + 2*y0) < (65)))) + 4*y0*y1 + ((65) * ((65) <= (2 + 2*y0)) + (2 + 2*y0) * ((2 + 2*y0) < (65))) + ((65) * ((65) <= (2 + 2*y1)) + (2 + 2*y1) * ((2 + 2*y1) < (65)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + 1024*x3 + 35840*y2), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/io/ciozhsh7obzlx2nluwm5qdb2ww4y77xlqppdlykkgmtv7btzvvwa.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_2 => avg_pool2d_2
# Graph fragment:
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%avg_pool2d, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_6 = async_compile.triton('triton_poi_fused_avg_pool2d_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex // 16
    x2 = (xindex % 16)
    y0 = (yindex % 3)
    y1 = yindex // 3
    x5 = xindex
    tmp0 = (-1) + 2*x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-33) + 2*x2 + 64*x3 + 1024*y0 + 35840*y1), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x2
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-32) + 2*x2 + 64*x3 + 1024*y0 + 35840*y1), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x2
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-31) + 2*x2 + 64*x3 + 1024*y0 + 35840*y1), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x3
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x2 + 64*x3 + 1024*y0 + 35840*y1), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x2 + 64*x3 + 1024*y0 + 35840*y1), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x2 + 64*x3 + 1024*y0 + 35840*y1), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x3
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (31 + 2*x2 + 64*x3 + 1024*y0 + 35840*y1), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (32 + 2*x2 + 64*x3 + 1024*y0 + 35840*y1), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (33 + 2*x2 + 64*x3 + 1024*y0 + 35840*y1), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x2) + ((-2)*x3) + ((33) * ((33) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (33)))*((33) * ((33) <= (2 + 2*x3)) + (2 + 2*x3) * ((2 + 2*x3) < (33))) + ((-2)*x2*((33) * ((33) <= (2 + 2*x3)) + (2 + 2*x3) * ((2 + 2*x3) < (33)))) + ((-2)*x3*((33) * ((33) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (33)))) + 4*x2*x3 + ((33) * ((33) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (33))) + ((33) * ((33) <= (2 + 2*x3)) + (2 + 2*x3) * ((2 + 2*x3) < (33)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y0 + 131*x5 + 33536*y1), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ue/cuef2gr7np3xg4rf6emvuvbyn2ty26uvb7wvse2l5balwvhc6hcv.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => gt, mul_3, where
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %add_1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/ih/cihstqvx65caegtin2ji2f55bemani6zbgnwusoiacmbb6vz4bkw.py
# Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   input_8 => add_5, mul_10, mul_9, sub_2
#   input_9 => gt_2, mul_11, where_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_5, 0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %add_5), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_5, %mul_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (x1 + 32*y0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr1 + (y2 + 1024*x1 + 35840*y3), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ra/craxc2guzbh7s7fwlzwrokajyznsugjdza4qp3cct7iz5vok3fts.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%where_2, %avg_pool2d], 1), kwargs = {})
triton_poi_fused_cat_9 = async_compile.triton('triton_poi_fused_cat_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 140
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 35)
    y1 = yindex // 35
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 35*x2 + 35840*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/v2/cv2tiyu5u2jqqqxabyv3yfon7hzadfhjlrisoz7ysxwfjpozvxwj.py
# Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   input_10 => add_7, mul_13, mul_14, sub_3
#   input_11 => gt_3, mul_15, where_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 0), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %add_7), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_7, %mul_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 35
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
    tmp0 = tl.load(in_ptr0 + (x1 + 35*y0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr1 + (y2 + 1024*x1 + 35840*y3), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kn/ckn4eetyrc7ravtexbf4x4a3v6epuv6kpctsro2fkejvfi4nweab.py
# Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   input_13 => add_9, mul_17, mul_18, sub_4
#   input_14 => gt_4, mul_19, where_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_33), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %add_9), kwargs = {})
#   %where_4 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %add_9, %mul_19), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/ij/ciju3j664ebtu4vwkhlmton5cgxpwl7awbvdd4aycsvi3s7chuvz.py
# Topologically Sorted Source Nodes: [joi_feat, joi_feat_1, joi_feat_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   joi_feat => cat_1
#   joi_feat_1 => add_11, mul_21, mul_22, sub_5
#   joi_feat_2 => gt_5, mul_23, where_5
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_4, %convolution_5], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_47), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_11, 0), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %add_11), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_11, %mul_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (64*x1 + ((-64) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp29 = tmp28 * tmp25
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(in_out_ptr0 + (x2), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/ve/cveawpr7dttsx4rala7l4uthk7jzqaa4lbhb3oumif3xfpxxkq6c.py
# Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_6, [-1, -2], True), kwargs = {})
triton_red_fused_mean_13 = async_compile.triton('triton_red_fused_mean_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_13(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 8192*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/us/cuszz5bvryliratbtpzjh35vooxhnlxi6hxzxuhz6nw4ap2t5dq7.py
# Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_6, [-1, -2], True), kwargs = {})
triton_per_fused_mean_14 = async_compile.triton('triton_per_fused_mean_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_14(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 128*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 256.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/al/calc5gjzvk6a3fm5gwt6535koop4a2wh3srumf2gewjnfhql4nlg.py
# Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_15 => add_tensor_23
#   input_16 => relu
# Graph fragment:
#   %add_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %primals_40), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_23,), kwargs = {})
triton_poi_fused_addmm_relu_15 = async_compile.triton('triton_poi_fused_addmm_relu_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4w/c4w24dfzoeqoh6fdnas5jbjgfeshee3iiuz2u7cdy6wn7vjf7tj4.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   out => mul_24
# Graph fragment:
#   %mul_24 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_6, %view_7), kwargs = {})
triton_poi_fused_mul_16 = async_compile.triton('triton_poi_fused_mul_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 64)
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/mz/cmzpqcla24w64mbtrrktc3yctyazt7wnece5tadi3kgqmlughcav.py
# Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   input_20 => add_13, mul_26, mul_27, sub_6
#   input_21 => gt_6, mul_28, where_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_49), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_55), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_13, 0), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %add_13), kwargs = {})
#   %where_6 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_13, %mul_28), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/65/c65fzsdmhbeld742keez72ivyptpeyoyohvrcuxtdrl6v6qxswen.py
# Topologically Sorted Source Nodes: [joi_feat_4, joi_feat_5], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   joi_feat_4 => cat_2
#   joi_feat_5 => add_15, mul_30, mul_31, sub_7
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_8, %convolution_9], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_2, %unsqueeze_57), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %unsqueeze_61), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_63), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 64, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (32*x1 + ((-32) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(out_ptr1 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/zg/czghvch46nwh5krqdxusjmzupatftgimexbo5wk6qduhc23pjtps.py
# Topologically Sorted Source Nodes: [joi_feat_6, adaptive_avg_pool2d_1], Original ATen: [aten._prelu_kernel, aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d_1 => mean_1
#   joi_feat_6 => gt_7, mul_32, where_7
# Graph fragment:
#   %gt_7 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_15, 0), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %add_15), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %add_15, %mul_32), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%where_7, [-1, -2], True), kwargs = {})
triton_red_fused__prelu_kernel_mean_19 = async_compile.triton('triton_red_fused__prelu_kernel_mean_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__prelu_kernel_mean_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__prelu_kernel_mean_19(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 8192*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = tmp3 * tmp0
        tmp5 = tl.where(tmp2, tmp0, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/og/cog454qrky5lv4al6tirfesrbiry6xjepyjpjthhgcog5hd6yldy.py
# Topologically Sorted Source Nodes: [joi_feat_6, out_1, x_3, cat_4], Original ATen: [aten._prelu_kernel, aten.mul, aten.add, aten.cat]
# Source node to ATen node mapping:
#   cat_4 => cat_4
#   joi_feat_6 => gt_7, mul_32, where_7
#   out_1 => mul_33
#   x_3 => add_16
# Graph fragment:
#   %gt_7 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_15, 0), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %add_15), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %add_15, %mul_32), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_7, %view_11), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %mul_33), kwargs = {})
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_21, %mul_24, %avg_pool2d_2], 1), kwargs = {})
triton_poi_fused__prelu_kernel_add_cat_mul_20 = async_compile.triton('triton_poi_fused__prelu_kernel_add_cat_mul_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_cat_mul_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_cat_mul_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = (xindex % 64)
    x2 = xindex // 16384
    x3 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_out_ptr0 + (x4), None)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp5 = tmp4 * tmp1
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(in_out_ptr0 + (x4), tmp10, None)
    tl.store(out_ptr0 + (x0 + 131*x3), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/gv/cgvw5mnc6bwg4wbpxumush2behhzslq3bbk2jr3yncccatikt42b.py
# Topologically Sorted Source Nodes: [joi_feat_9, out_2, x_4], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
# Source node to ATen node mapping:
#   joi_feat_9 => gt_9, mul_41, where_9
#   out_2 => mul_42
#   x_4 => add_21
# Graph fragment:
#   %gt_9 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_20, 0), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %add_20), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %add_20, %mul_41), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_9, %view_15), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %mul_42), kwargs = {})
triton_poi_fused__prelu_kernel_add_mul_21 = async_compile.triton('triton_poi_fused__prelu_kernel_add_mul_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_mul_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_mul_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 64)
    x2 = xindex // 16384
    x4 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp5 = tmp4 * tmp1
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(out_ptr0 + (x0 + 131*x4), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/ap/caprzmeyxnstor3taxev322jmu34u442jkilf3zykmu7pd2zt7yj.py
# Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   input_33 => add_23, mul_44, mul_45, sub_10
#   input_34 => gt_10, mul_46, where_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_4, %unsqueeze_81), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %unsqueeze_85), kwargs = {})
#   %add_23 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %unsqueeze_87), kwargs = {})
#   %gt_10 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_23, 0), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, %add_23), kwargs = {})
#   %where_10 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_10, %add_23, %mul_46), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 131
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = (yindex % 256)
    y3 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x1 + 131*y0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr1 + (y2 + 256*x1 + 33536*y3), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b6/cb6fesrwrkk6vjdmteoszdj4illx4oooppvrhpvbvlg4an22karz.py
# Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_35 => convolution_13
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_10, %primals_82, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_23 = async_compile.triton('triton_poi_fused_convolution_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 131)
    y1 = yindex // 131
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 131*x2 + 33536*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/eq/ceqfbgbwdpjabau2vk2mpriqatd23mkavig6wpgq6x7obf2rj5bo.py
# Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   input_36 => add_25, mul_48, mul_49, sub_11
#   input_37 => gt_11, mul_50, where_11
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_89), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %unsqueeze_93), kwargs = {})
#   %add_25 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_95), kwargs = {})
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_25, 0), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %add_25), kwargs = {})
#   %where_11 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_11, %add_25, %mul_50), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/j7/cj7ziebnt2k2d3m6khv5t43bb3ckya36pq4bnta3e32gefsskm5i.py
# Topologically Sorted Source Nodes: [joi_feat_10, joi_feat_11, joi_feat_12], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   joi_feat_10 => cat_5
#   joi_feat_11 => add_27, mul_52, mul_53, sub_12
#   joi_feat_12 => gt_12, mul_54, where_12
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_14, %convolution_15], 1), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_5, %unsqueeze_97), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_101), kwargs = {})
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_103), kwargs = {})
#   %gt_12 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_27, 0), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_18, %add_27), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %add_27, %mul_54), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 256, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (128*x1 + ((-128) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp29 = tmp28 * tmp25
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(in_out_ptr0 + (x2), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/rf/crfmzlfh45qtkxcnthkorkr4skgjddz47eznp3nbhq5lt3d4gi4x.py
# Topologically Sorted Source Nodes: [adaptive_avg_pool2d_3], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d_3 => mean_3
# Graph fragment:
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_16, [-1, -2], True), kwargs = {})
triton_per_fused_mean_26 = async_compile.triton('triton_per_fused_mean_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_26(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r2 + 8192*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d7/cd7agsa2uij7nf2qkd3ddzm3grpnkpt54h2u4o7s3e3nuhynjpk7.py
# Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   out_3 => mul_55
# Graph fragment:
#   %mul_55 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_16, %view_20), kwargs = {})
triton_poi_fused_mul_27 = async_compile.triton('triton_poi_fused_mul_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/gu/cguqarz2kcqrhawxsjb3eyk22opbrwapgoko7engiwgvothbz33a.py
# Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   input_43 => add_29, mul_57, mul_58, sub_13
#   input_44 => gt_13, mul_59, where_13
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_105), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %unsqueeze_109), kwargs = {})
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_58, %unsqueeze_111), kwargs = {})
#   %gt_13 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_29, 0), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, %add_29), kwargs = {})
#   %where_13 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_13, %add_29, %mul_59), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/74/c74653d3utj3jg6fzv35rhltxee4lma3bhzocg26piklhij33xvz.py
# Topologically Sorted Source Nodes: [joi_feat_14, joi_feat_15], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   joi_feat_14 => cat_6
#   joi_feat_15 => add_31, mul_61, mul_62, sub_14
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_18, %convolution_19], 1), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_6, %unsqueeze_113), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_117), kwargs = {})
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_119), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (64*x1 + ((-64) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(out_ptr1 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/kt/ckt3nnizlq52de5hoj25aocyn6bqigv3ctv36dtoycghno7gnm3t.py
# Topologically Sorted Source Nodes: [joi_feat_16, adaptive_avg_pool2d_4], Original ATen: [aten._prelu_kernel, aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d_4 => mean_4
#   joi_feat_16 => gt_14, mul_63, where_14
# Graph fragment:
#   %gt_14 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_31, 0), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %add_31), kwargs = {})
#   %where_14 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %add_31, %mul_63), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%where_14, [-1, -2], True), kwargs = {})
triton_per_fused__prelu_kernel_mean_30 = async_compile.triton('triton_per_fused__prelu_kernel_mean_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__prelu_kernel_mean_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__prelu_kernel_mean_30(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r2 + 8192*x1), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = 64.0
    tmp11 = tmp9 / tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ys/cys3s2q2ttzgzatunalpauidcogndqnz7le2g6qcnmeldobngvfu.py
# Topologically Sorted Source Nodes: [joi_feat_16, out_4, x_5], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
# Source node to ATen node mapping:
#   joi_feat_16 => gt_14, mul_63, where_14
#   out_4 => mul_64
#   x_5 => add_32
# Graph fragment:
#   %gt_14 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_31, 0), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %add_31), kwargs = {})
#   %where_14 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %add_31, %mul_63), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_14, %view_24), kwargs = {})
#   %add_32 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %mul_64), kwargs = {})
triton_poi_fused__prelu_kernel_add_mul_31 = async_compile.triton('triton_poi_fused__prelu_kernel_add_mul_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_mul_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_mul_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp5 = tmp4 * tmp1
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/6s/c6snqral2cga6kcvwcun52ivdbwdiym73i3wjdoctbxvkexq4hdy.py
# Topologically Sorted Source Nodes: [cat_26, input_182, input_183], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   cat_26 => cat_26
#   input_182 => add_129, mul_237, mul_238, sub_53
#   input_183 => gt_53, mul_239, where_53
# Graph fragment:
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_55, %add_127], 1), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_26, %unsqueeze_425), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_237, %unsqueeze_429), kwargs = {})
#   %add_129 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %unsqueeze_431), kwargs = {})
#   %gt_53 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_129, 0), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_101, %add_129), kwargs = {})
#   %where_53 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_53, %add_129, %mul_239), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = yindex // 64
    y0 = (yindex % 64)
    tmp23 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*y3 + (x2)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 256, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (128*y3 + ((-128) + x2)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (128*y3 + ((-128) + x2)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = 0.0
    tmp12 = tmp10 > tmp11
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to((-128) + x2, [XBLOCK, YBLOCK])), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp13 * tmp10
    tmp15 = tl.where(tmp12, tmp10, tmp14)
    tmp16 = tl.load(in_ptr4 + (128*y1 + ((-128) + x2)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = tmp9 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1, 1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = 0.0
    tmp39 = tmp37 > tmp38
    tmp41 = tmp40 * tmp37
    tmp42 = tl.where(tmp39, tmp37, tmp41)
    tl.store(out_ptr0 + (x2 + 256*y3), tmp22, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 64*x2 + 16384*y1), tmp42, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, ), (1, ))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (32, ), (1, ))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (35, ), (1, ))
    assert_size_stride(primals_21, (35, ), (1, ))
    assert_size_stride(primals_22, (35, ), (1, ))
    assert_size_stride(primals_23, (35, ), (1, ))
    assert_size_stride(primals_24, (35, ), (1, ))
    assert_size_stride(primals_25, (64, 35, 3, 3), (315, 9, 3, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_39, (8, 64), (64, 1))
    assert_size_stride(primals_40, (8, ), (1, ))
    assert_size_stride(primals_41, (64, 8), (8, 1))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (32, ), (1, ))
    assert_size_stride(primals_47, (32, ), (1, ))
    assert_size_stride(primals_48, (32, ), (1, ))
    assert_size_stride(primals_49, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_50, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (64, ), (1, ))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (8, 64), (64, 1))
    assert_size_stride(primals_57, (8, ), (1, ))
    assert_size_stride(primals_58, (64, 8), (8, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (32, ), (1, ))
    assert_size_stride(primals_63, (32, ), (1, ))
    assert_size_stride(primals_64, (32, ), (1, ))
    assert_size_stride(primals_65, (32, ), (1, ))
    assert_size_stride(primals_66, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_67, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (8, 64), (64, 1))
    assert_size_stride(primals_74, (8, ), (1, ))
    assert_size_stride(primals_75, (64, 8), (8, 1))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (131, ), (1, ))
    assert_size_stride(primals_78, (131, ), (1, ))
    assert_size_stride(primals_79, (131, ), (1, ))
    assert_size_stride(primals_80, (131, ), (1, ))
    assert_size_stride(primals_81, (131, ), (1, ))
    assert_size_stride(primals_82, (128, 131, 3, 3), (1179, 9, 3, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_89, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (256, ), (1, ))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_96, (8, 128), (128, 1))
    assert_size_stride(primals_97, (8, ), (1, ))
    assert_size_stride(primals_98, (128, 8), (8, 1))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (64, ), (1, ))
    assert_size_stride(primals_103, (64, ), (1, ))
    assert_size_stride(primals_104, (64, ), (1, ))
    assert_size_stride(primals_105, (64, ), (1, ))
    assert_size_stride(primals_106, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_113, (8, 128), (128, 1))
    assert_size_stride(primals_114, (8, ), (1, ))
    assert_size_stride(primals_115, (128, 8), (8, 1))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_118, (64, ), (1, ))
    assert_size_stride(primals_119, (64, ), (1, ))
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (64, ), (1, ))
    assert_size_stride(primals_123, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_124, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (8, 128), (128, 1))
    assert_size_stride(primals_131, (8, ), (1, ))
    assert_size_stride(primals_132, (128, 8), (8, 1))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_135, (64, ), (1, ))
    assert_size_stride(primals_136, (64, ), (1, ))
    assert_size_stride(primals_137, (64, ), (1, ))
    assert_size_stride(primals_138, (64, ), (1, ))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_141, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (8, 128), (128, 1))
    assert_size_stride(primals_148, (8, ), (1, ))
    assert_size_stride(primals_149, (128, 8), (8, 1))
    assert_size_stride(primals_150, (128, ), (1, ))
    assert_size_stride(primals_151, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_152, (64, ), (1, ))
    assert_size_stride(primals_153, (64, ), (1, ))
    assert_size_stride(primals_154, (64, ), (1, ))
    assert_size_stride(primals_155, (64, ), (1, ))
    assert_size_stride(primals_156, (64, ), (1, ))
    assert_size_stride(primals_157, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_158, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (8, 128), (128, 1))
    assert_size_stride(primals_165, (8, ), (1, ))
    assert_size_stride(primals_166, (128, 8), (8, 1))
    assert_size_stride(primals_167, (128, ), (1, ))
    assert_size_stride(primals_168, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (64, ), (1, ))
    assert_size_stride(primals_171, (64, ), (1, ))
    assert_size_stride(primals_172, (64, ), (1, ))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_175, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_176, (128, ), (1, ))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (8, 128), (128, 1))
    assert_size_stride(primals_182, (8, ), (1, ))
    assert_size_stride(primals_183, (128, 8), (8, 1))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (64, ), (1, ))
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (64, ), (1, ))
    assert_size_stride(primals_191, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_192, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (128, ), (1, ))
    assert_size_stride(primals_194, (128, ), (1, ))
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (128, ), (1, ))
    assert_size_stride(primals_198, (8, 128), (128, 1))
    assert_size_stride(primals_199, (8, ), (1, ))
    assert_size_stride(primals_200, (128, 8), (8, 1))
    assert_size_stride(primals_201, (128, ), (1, ))
    assert_size_stride(primals_202, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_203, (64, ), (1, ))
    assert_size_stride(primals_204, (64, ), (1, ))
    assert_size_stride(primals_205, (64, ), (1, ))
    assert_size_stride(primals_206, (64, ), (1, ))
    assert_size_stride(primals_207, (64, ), (1, ))
    assert_size_stride(primals_208, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_209, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_210, (128, ), (1, ))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (128, ), (1, ))
    assert_size_stride(primals_214, (128, ), (1, ))
    assert_size_stride(primals_215, (8, 128), (128, 1))
    assert_size_stride(primals_216, (8, ), (1, ))
    assert_size_stride(primals_217, (128, 8), (8, 1))
    assert_size_stride(primals_218, (128, ), (1, ))
    assert_size_stride(primals_219, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_221, (64, ), (1, ))
    assert_size_stride(primals_222, (64, ), (1, ))
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, ), (1, ))
    assert_size_stride(primals_225, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_226, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_227, (128, ), (1, ))
    assert_size_stride(primals_228, (128, ), (1, ))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, ), (1, ))
    assert_size_stride(primals_231, (128, ), (1, ))
    assert_size_stride(primals_232, (8, 128), (128, 1))
    assert_size_stride(primals_233, (8, ), (1, ))
    assert_size_stride(primals_234, (128, 8), (8, 1))
    assert_size_stride(primals_235, (128, ), (1, ))
    assert_size_stride(primals_236, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_237, (64, ), (1, ))
    assert_size_stride(primals_238, (64, ), (1, ))
    assert_size_stride(primals_239, (64, ), (1, ))
    assert_size_stride(primals_240, (64, ), (1, ))
    assert_size_stride(primals_241, (64, ), (1, ))
    assert_size_stride(primals_242, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_244, (128, ), (1, ))
    assert_size_stride(primals_245, (128, ), (1, ))
    assert_size_stride(primals_246, (128, ), (1, ))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, ), (1, ))
    assert_size_stride(primals_249, (8, 128), (128, 1))
    assert_size_stride(primals_250, (8, ), (1, ))
    assert_size_stride(primals_251, (128, 8), (8, 1))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_254, (64, ), (1, ))
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (64, ), (1, ))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_260, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (128, ), (1, ))
    assert_size_stride(primals_265, (128, ), (1, ))
    assert_size_stride(primals_266, (8, 128), (128, 1))
    assert_size_stride(primals_267, (8, ), (1, ))
    assert_size_stride(primals_268, (128, 8), (8, 1))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_271, (64, ), (1, ))
    assert_size_stride(primals_272, (64, ), (1, ))
    assert_size_stride(primals_273, (64, ), (1, ))
    assert_size_stride(primals_274, (64, ), (1, ))
    assert_size_stride(primals_275, (64, ), (1, ))
    assert_size_stride(primals_276, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_277, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_278, (128, ), (1, ))
    assert_size_stride(primals_279, (128, ), (1, ))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (128, ), (1, ))
    assert_size_stride(primals_283, (8, 128), (128, 1))
    assert_size_stride(primals_284, (8, ), (1, ))
    assert_size_stride(primals_285, (128, 8), (8, 1))
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_288, (64, ), (1, ))
    assert_size_stride(primals_289, (64, ), (1, ))
    assert_size_stride(primals_290, (64, ), (1, ))
    assert_size_stride(primals_291, (64, ), (1, ))
    assert_size_stride(primals_292, (64, ), (1, ))
    assert_size_stride(primals_293, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_294, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (128, ), (1, ))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (128, ), (1, ))
    assert_size_stride(primals_300, (8, 128), (128, 1))
    assert_size_stride(primals_301, (8, ), (1, ))
    assert_size_stride(primals_302, (128, 8), (8, 1))
    assert_size_stride(primals_303, (128, ), (1, ))
    assert_size_stride(primals_304, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_305, (64, ), (1, ))
    assert_size_stride(primals_306, (64, ), (1, ))
    assert_size_stride(primals_307, (64, ), (1, ))
    assert_size_stride(primals_308, (64, ), (1, ))
    assert_size_stride(primals_309, (64, ), (1, ))
    assert_size_stride(primals_310, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_311, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_312, (128, ), (1, ))
    assert_size_stride(primals_313, (128, ), (1, ))
    assert_size_stride(primals_314, (128, ), (1, ))
    assert_size_stride(primals_315, (128, ), (1, ))
    assert_size_stride(primals_316, (128, ), (1, ))
    assert_size_stride(primals_317, (8, 128), (128, 1))
    assert_size_stride(primals_318, (8, ), (1, ))
    assert_size_stride(primals_319, (128, 8), (8, 1))
    assert_size_stride(primals_320, (128, ), (1, ))
    assert_size_stride(primals_321, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_322, (64, ), (1, ))
    assert_size_stride(primals_323, (64, ), (1, ))
    assert_size_stride(primals_324, (64, ), (1, ))
    assert_size_stride(primals_325, (64, ), (1, ))
    assert_size_stride(primals_326, (64, ), (1, ))
    assert_size_stride(primals_327, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_328, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_329, (128, ), (1, ))
    assert_size_stride(primals_330, (128, ), (1, ))
    assert_size_stride(primals_331, (128, ), (1, ))
    assert_size_stride(primals_332, (128, ), (1, ))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (8, 128), (128, 1))
    assert_size_stride(primals_335, (8, ), (1, ))
    assert_size_stride(primals_336, (128, 8), (8, 1))
    assert_size_stride(primals_337, (128, ), (1, ))
    assert_size_stride(primals_338, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_339, (64, ), (1, ))
    assert_size_stride(primals_340, (64, ), (1, ))
    assert_size_stride(primals_341, (64, ), (1, ))
    assert_size_stride(primals_342, (64, ), (1, ))
    assert_size_stride(primals_343, (64, ), (1, ))
    assert_size_stride(primals_344, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_345, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_346, (128, ), (1, ))
    assert_size_stride(primals_347, (128, ), (1, ))
    assert_size_stride(primals_348, (128, ), (1, ))
    assert_size_stride(primals_349, (128, ), (1, ))
    assert_size_stride(primals_350, (128, ), (1, ))
    assert_size_stride(primals_351, (8, 128), (128, 1))
    assert_size_stride(primals_352, (8, ), (1, ))
    assert_size_stride(primals_353, (128, 8), (8, 1))
    assert_size_stride(primals_354, (128, ), (1, ))
    assert_size_stride(primals_355, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_356, (64, ), (1, ))
    assert_size_stride(primals_357, (64, ), (1, ))
    assert_size_stride(primals_358, (64, ), (1, ))
    assert_size_stride(primals_359, (64, ), (1, ))
    assert_size_stride(primals_360, (64, ), (1, ))
    assert_size_stride(primals_361, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_362, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_363, (128, ), (1, ))
    assert_size_stride(primals_364, (128, ), (1, ))
    assert_size_stride(primals_365, (128, ), (1, ))
    assert_size_stride(primals_366, (128, ), (1, ))
    assert_size_stride(primals_367, (128, ), (1, ))
    assert_size_stride(primals_368, (8, 128), (128, 1))
    assert_size_stride(primals_369, (8, ), (1, ))
    assert_size_stride(primals_370, (128, 8), (8, 1))
    assert_size_stride(primals_371, (128, ), (1, ))
    assert_size_stride(primals_372, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_373, (64, ), (1, ))
    assert_size_stride(primals_374, (64, ), (1, ))
    assert_size_stride(primals_375, (64, ), (1, ))
    assert_size_stride(primals_376, (64, ), (1, ))
    assert_size_stride(primals_377, (64, ), (1, ))
    assert_size_stride(primals_378, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_379, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_380, (128, ), (1, ))
    assert_size_stride(primals_381, (128, ), (1, ))
    assert_size_stride(primals_382, (128, ), (1, ))
    assert_size_stride(primals_383, (128, ), (1, ))
    assert_size_stride(primals_384, (128, ), (1, ))
    assert_size_stride(primals_385, (8, 128), (128, 1))
    assert_size_stride(primals_386, (8, ), (1, ))
    assert_size_stride(primals_387, (128, 8), (8, 1))
    assert_size_stride(primals_388, (128, ), (1, ))
    assert_size_stride(primals_389, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_390, (64, ), (1, ))
    assert_size_stride(primals_391, (64, ), (1, ))
    assert_size_stride(primals_392, (64, ), (1, ))
    assert_size_stride(primals_393, (64, ), (1, ))
    assert_size_stride(primals_394, (64, ), (1, ))
    assert_size_stride(primals_395, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_396, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_397, (128, ), (1, ))
    assert_size_stride(primals_398, (128, ), (1, ))
    assert_size_stride(primals_399, (128, ), (1, ))
    assert_size_stride(primals_400, (128, ), (1, ))
    assert_size_stride(primals_401, (128, ), (1, ))
    assert_size_stride(primals_402, (8, 128), (128, 1))
    assert_size_stride(primals_403, (8, ), (1, ))
    assert_size_stride(primals_404, (128, 8), (8, 1))
    assert_size_stride(primals_405, (128, ), (1, ))
    assert_size_stride(primals_406, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_407, (64, ), (1, ))
    assert_size_stride(primals_408, (64, ), (1, ))
    assert_size_stride(primals_409, (64, ), (1, ))
    assert_size_stride(primals_410, (64, ), (1, ))
    assert_size_stride(primals_411, (64, ), (1, ))
    assert_size_stride(primals_412, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_413, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_414, (128, ), (1, ))
    assert_size_stride(primals_415, (128, ), (1, ))
    assert_size_stride(primals_416, (128, ), (1, ))
    assert_size_stride(primals_417, (128, ), (1, ))
    assert_size_stride(primals_418, (128, ), (1, ))
    assert_size_stride(primals_419, (8, 128), (128, 1))
    assert_size_stride(primals_420, (8, ), (1, ))
    assert_size_stride(primals_421, (128, 8), (8, 1))
    assert_size_stride(primals_422, (128, ), (1, ))
    assert_size_stride(primals_423, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_424, (64, ), (1, ))
    assert_size_stride(primals_425, (64, ), (1, ))
    assert_size_stride(primals_426, (64, ), (1, ))
    assert_size_stride(primals_427, (64, ), (1, ))
    assert_size_stride(primals_428, (64, ), (1, ))
    assert_size_stride(primals_429, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_430, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_431, (128, ), (1, ))
    assert_size_stride(primals_432, (128, ), (1, ))
    assert_size_stride(primals_433, (128, ), (1, ))
    assert_size_stride(primals_434, (128, ), (1, ))
    assert_size_stride(primals_435, (128, ), (1, ))
    assert_size_stride(primals_436, (8, 128), (128, 1))
    assert_size_stride(primals_437, (8, ), (1, ))
    assert_size_stride(primals_438, (128, 8), (8, 1))
    assert_size_stride(primals_439, (128, ), (1, ))
    assert_size_stride(primals_440, (256, ), (1, ))
    assert_size_stride(primals_441, (256, ), (1, ))
    assert_size_stride(primals_442, (256, ), (1, ))
    assert_size_stride(primals_443, (256, ), (1, ))
    assert_size_stride(primals_444, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_8, buf2, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_8
        buf3 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_14, buf3, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_14
        buf4 = empty_strided_cuda((64, 35, 3, 3), (315, 1, 105, 35), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_25, buf4, 2240, 9, grid=grid(2240, 9), stream=stream0)
        del primals_25
        buf5 = empty_strided_cuda((128, 131, 3, 3), (1179, 1, 393, 131), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_82, buf5, 16768, 9, grid=grid(16768, 9), stream=stream0)
        del primals_82
        buf17 = empty_strided_cuda((4, 35, 32, 32), (35840, 1024, 32, 1), torch.float32)
        buf6 = reinterpret_tensor(buf17, (4, 3, 32, 32), (35840, 1024, 32, 1), 32768)  # alias
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_5.run(buf0, buf6, 4096, 3, grid=grid(4096, 3), stream=stream0)
        buf67 = empty_strided_cuda((4, 131, 16, 16), (33536, 1, 2096, 131), torch.float32)
        buf7 = reinterpret_tensor(buf67, (4, 3, 16, 16), (33536, 1, 2096, 131), 128)  # alias
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_6.run(buf6, buf7, 12, 256, grid=grid(12, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf9 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_7.run(buf10, buf8, primals_3, primals_4, primals_5, primals_6, primals_7, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf12 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_7.run(buf13, buf11, primals_9, primals_10, primals_11, primals_12, primals_13, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf16 = reinterpret_tensor(buf17, (4, 32, 32, 32), (35840, 1024, 32, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_8.run(buf14, primals_15, primals_16, primals_17, primals_18, primals_19, buf16, 4096, 32, grid=grid(4096, 32), stream=stream0)
        buf18 = empty_strided_cuda((4, 35, 32, 32), (35840, 1, 1120, 35), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf17, buf18, 140, 1024, grid=grid(140, 1024), stream=stream0)
        del buf16
        del buf6
        buf20 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10.run(buf18, primals_20, primals_21, primals_22, primals_23, primals_24, buf20, 4096, 35, grid=grid(4096, 35), stream=stream0)
        buf21 = empty_strided_cuda((4, 35, 32, 32), (35840, 1, 1120, 35), torch.float32)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf20, buf21, 140, 1024, grid=grid(140, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 64, 16, 16), (16384, 1, 1024, 64))
        del buf21
        buf23 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_11.run(buf24, buf22, primals_26, primals_27, primals_28, primals_29, primals_30, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [loc], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf25, (4, 64, 16, 16), (16384, 1, 1024, 64))
        # Topologically Sorted Source Nodes: [sur], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf24, primals_32, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf26, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf27 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf28 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [joi_feat, joi_feat_1, joi_feat_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_12.run(buf29, buf25, buf26, primals_33, primals_34, primals_35, primals_36, primals_37, buf27, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [joi_feat_3], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf31 = empty_strided_cuda((4, 64, 1, 1, 2), (128, 1, 512, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_13.run(buf30, buf31, 512, 128, grid=grid(512), stream=stream0)
        buf32 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_14.run(buf33, buf31, 256, 2, grid=grid(256), stream=stream0)
        buf34 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf33, (4, 64), (64, 1), 0), reinterpret_tensor(primals_39, (64, 8), (1, 64), 0), out=buf34)
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf35, primals_40, 32, grid=grid(32), stream=stream0)
        del primals_40
        buf36 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_42, buf35, reinterpret_tensor(primals_41, (8, 64), (1, 8), 0), alpha=1, beta=1, out=buf36)
        del primals_42
        buf37 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_16.run(buf30, buf36, buf37, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf39 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.float32)
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_17.run(buf40, buf38, primals_44, primals_45, primals_46, primals_47, primals_48, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_1], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf41, (4, 32, 16, 16), (8192, 1, 512, 32))
        # Topologically Sorted Source Nodes: [sur_1], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf40, primals_50, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf42, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf43 = buf25; del buf25  # reuse
        buf44 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_4, joi_feat_5], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_18.run(buf41, buf42, primals_51, primals_52, primals_53, primals_54, buf43, buf44, 65536, grid=grid(65536), stream=stream0)
        buf45 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_6, adaptive_avg_pool2d_1], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused__prelu_kernel_mean_19.run(buf44, primals_55, buf45, 512, 128, grid=grid(512), stream=stream0)
        buf46 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_6, adaptive_avg_pool2d_1], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_14.run(buf47, buf45, 256, 2, grid=grid(256), stream=stream0)
        buf48 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf47, (4, 64), (64, 1), 0), reinterpret_tensor(primals_56, (64, 8), (1, 64), 0), out=buf48)
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf49, primals_57, 32, grid=grid(32), stream=stream0)
        del primals_57
        buf50 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_59, buf49, reinterpret_tensor(primals_58, (8, 64), (1, 8), 0), alpha=1, beta=1, out=buf50)
        del primals_59
        buf51 = buf44; del buf44  # reuse
        buf66 = reinterpret_tensor(buf67, (4, 64, 16, 16), (33536, 1, 2096, 131), 64)  # alias
        # Topologically Sorted Source Nodes: [joi_feat_6, out_1, x_3, cat_4], Original ATen: [aten._prelu_kernel, aten.mul, aten.add, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_cat_mul_20.run(buf51, buf37, primals_55, buf50, buf66, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf53 = buf42; del buf42  # reuse
        buf54 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_17.run(buf54, buf52, primals_61, primals_62, primals_63, primals_64, primals_65, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_2], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf55, (4, 32, 16, 16), (8192, 1, 512, 32))
        # Topologically Sorted Source Nodes: [sur_2], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf54, primals_67, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf56, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf57 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf58 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_7, joi_feat_8], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_18.run(buf55, buf56, primals_68, primals_69, primals_70, primals_71, buf57, buf58, 65536, grid=grid(65536), stream=stream0)
        buf59 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_9, adaptive_avg_pool2d_2], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused__prelu_kernel_mean_19.run(buf58, primals_72, buf59, 512, 128, grid=grid(512), stream=stream0)
        buf60 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_9, adaptive_avg_pool2d_2], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_14.run(buf61, buf59, 256, 2, grid=grid(256), stream=stream0)
        buf62 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf61, (4, 64), (64, 1), 0), reinterpret_tensor(primals_73, (64, 8), (1, 64), 0), out=buf62)
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf63, primals_74, 32, grid=grid(32), stream=stream0)
        del primals_74
        buf64 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_76, buf63, reinterpret_tensor(primals_75, (8, 64), (1, 8), 0), alpha=1, beta=1, out=buf64)
        del primals_76
        buf65 = reinterpret_tensor(buf67, (4, 64, 16, 16), (33536, 1, 2096, 131), 0)  # alias
        # Topologically Sorted Source Nodes: [joi_feat_9, out_2, x_4], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_21.run(buf51, buf58, primals_72, buf64, buf65, 65536, grid=grid(65536), stream=stream0)
        buf69 = empty_strided_cuda((4, 131, 16, 16), (33536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22.run(buf67, primals_77, primals_78, primals_79, primals_80, primals_81, buf69, 1024, 131, grid=grid(1024, 131), stream=stream0)
        buf70 = empty_strided_cuda((4, 131, 16, 16), (33536, 1, 2096, 131), torch.float32)
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_23.run(buf69, buf70, 524, 256, grid=grid(524, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 128, 8, 8), (8192, 1, 1024, 128))
        del buf70
        buf72 = reinterpret_tensor(buf56, (4, 128, 8, 8), (8192, 1, 1024, 128), 0); del buf56  # reuse
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_24.run(buf73, buf71, primals_83, primals_84, primals_85, primals_86, primals_87, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_3], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf74, (4, 128, 8, 8), (8192, 1, 1024, 128))
        # Topologically Sorted Source Nodes: [sur_3], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf73, primals_89, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf75, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf76 = reinterpret_tensor(buf58, (4, 256, 8, 8), (16384, 1, 2048, 256), 0); del buf58  # reuse
        buf77 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_10, joi_feat_11, joi_feat_12], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_25.run(buf78, buf74, buf75, primals_90, primals_91, primals_92, primals_93, primals_94, buf76, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [joi_feat_13], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_95, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf80 = reinterpret_tensor(buf59, (4, 128, 1, 1), (128, 1, 512, 512), 0); del buf59  # reuse
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [adaptive_avg_pool2d_3], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_26.run(buf81, buf79, 512, 64, grid=grid(512), stream=stream0)
        buf82 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf81, (4, 128), (128, 1), 0), reinterpret_tensor(primals_96, (128, 8), (1, 128), 0), out=buf82)
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf83, primals_97, 32, grid=grid(32), stream=stream0)
        del primals_97
        buf84 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_99, buf83, reinterpret_tensor(primals_98, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf84)
        del primals_99
        buf85 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_27.run(buf79, buf84, buf85, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf87 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        buf88 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf88, buf86, primals_101, primals_102, primals_103, primals_104, primals_105, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_4], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf89, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_4], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf88, primals_107, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf90, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf91 = buf74; del buf74  # reuse
        buf92 = reinterpret_tensor(buf55, (4, 128, 8, 8), (8192, 1, 1024, 128), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_14, joi_feat_15], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf89, buf90, primals_108, primals_109, primals_110, primals_111, buf91, buf92, 32768, grid=grid(32768), stream=stream0)
        del buf89
        buf93 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_16, adaptive_avg_pool2d_4], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf94, buf92, primals_112, 512, 64, grid=grid(512), stream=stream0)
        buf95 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf94, (4, 128), (128, 1), 0), reinterpret_tensor(primals_113, (128, 8), (1, 128), 0), out=buf95)
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf96, primals_114, 32, grid=grid(32), stream=stream0)
        del primals_114
        buf97 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_116, buf96, reinterpret_tensor(primals_115, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf97)
        del primals_116
        buf98 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_16, out_4, x_5], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf98, buf85, primals_112, buf97, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf100 = buf90; del buf90  # reuse
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [input_50, input_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf101, buf99, primals_118, primals_119, primals_120, primals_121, primals_122, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_5], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf102, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_5], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf101, primals_124, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf103, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf104 = reinterpret_tensor(buf41, (4, 128, 8, 8), (8192, 1, 1024, 128), 0); del buf41  # reuse
        buf105 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_17, joi_feat_18], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf102, buf103, primals_125, primals_126, primals_127, primals_128, buf104, buf105, 32768, grid=grid(32768), stream=stream0)
        del buf102
        buf106 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_19, adaptive_avg_pool2d_5], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf107, buf105, primals_129, 512, 64, grid=grid(512), stream=stream0)
        buf108 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf107, (4, 128), (128, 1), 0), reinterpret_tensor(primals_130, (128, 8), (1, 128), 0), out=buf108)
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf109, primals_131, 32, grid=grid(32), stream=stream0)
        del primals_131
        buf110 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_133, buf109, reinterpret_tensor(primals_132, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf110)
        del primals_133
        buf111 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_19, out_5, x_6], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf111, buf98, primals_129, buf110, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf113 = buf103; del buf103  # reuse
        buf114 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf114, buf112, primals_135, primals_136, primals_137, primals_138, primals_139, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_6], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_140, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf115, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_6], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf114, primals_141, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf116, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf117 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf118 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_20, joi_feat_21], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf115, buf116, primals_142, primals_143, primals_144, primals_145, buf117, buf118, 32768, grid=grid(32768), stream=stream0)
        del buf115
        buf119 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf120 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_22, adaptive_avg_pool2d_6], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf120, buf118, primals_146, 512, 64, grid=grid(512), stream=stream0)
        buf121 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf120, (4, 128), (128, 1), 0), reinterpret_tensor(primals_147, (128, 8), (1, 128), 0), out=buf121)
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [input_59, input_60], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf122, primals_148, 32, grid=grid(32), stream=stream0)
        del primals_148
        buf123 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_150, buf122, reinterpret_tensor(primals_149, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf123)
        del primals_150
        buf124 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_22, out_6, x_7], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf124, buf111, primals_146, buf123, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf126 = buf116; del buf116  # reuse
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [input_64, input_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf127, buf125, primals_152, primals_153, primals_154, primals_155, primals_156, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_7], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf128, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_7], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf127, primals_158, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf129, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf130 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf131 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_23, joi_feat_24], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf128, buf129, primals_159, primals_160, primals_161, primals_162, buf130, buf131, 32768, grid=grid(32768), stream=stream0)
        del buf128
        buf132 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf133 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_25, adaptive_avg_pool2d_7], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf133, buf131, primals_163, 512, 64, grid=grid(512), stream=stream0)
        buf134 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf133, (4, 128), (128, 1), 0), reinterpret_tensor(primals_164, (128, 8), (1, 128), 0), out=buf134)
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [input_66, input_67], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf135, primals_165, 32, grid=grid(32), stream=stream0)
        del primals_165
        buf136 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_167, buf135, reinterpret_tensor(primals_166, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf136)
        del primals_167
        buf137 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_25, out_7, x_8], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf137, buf124, primals_163, buf136, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf139 = buf129; del buf129  # reuse
        buf140 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [input_71, input_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf140, buf138, primals_169, primals_170, primals_171, primals_172, primals_173, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_8], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf141, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_8], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf140, primals_175, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf142, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf143 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf144 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_26, joi_feat_27], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf141, buf142, primals_176, primals_177, primals_178, primals_179, buf143, buf144, 32768, grid=grid(32768), stream=stream0)
        del buf141
        buf145 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf146 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_28, adaptive_avg_pool2d_8], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf146, buf144, primals_180, 512, 64, grid=grid(512), stream=stream0)
        buf147 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf146, (4, 128), (128, 1), 0), reinterpret_tensor(primals_181, (128, 8), (1, 128), 0), out=buf147)
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [input_73, input_74], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf148, primals_182, 32, grid=grid(32), stream=stream0)
        del primals_182
        buf149 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_184, buf148, reinterpret_tensor(primals_183, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf149)
        del primals_184
        buf150 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_28, out_8, x_9], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf150, buf137, primals_180, buf149, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf152 = buf142; del buf142  # reuse
        buf153 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [input_78, input_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf153, buf151, primals_186, primals_187, primals_188, primals_189, primals_190, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_9], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf154, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_9], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf153, primals_192, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf155, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf156 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf157 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_29, joi_feat_30], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf154, buf155, primals_193, primals_194, primals_195, primals_196, buf156, buf157, 32768, grid=grid(32768), stream=stream0)
        del buf154
        buf158 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_31, adaptive_avg_pool2d_9], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf159, buf157, primals_197, 512, 64, grid=grid(512), stream=stream0)
        buf160 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf159, (4, 128), (128, 1), 0), reinterpret_tensor(primals_198, (128, 8), (1, 128), 0), out=buf160)
        buf161 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf161, primals_199, 32, grid=grid(32), stream=stream0)
        del primals_199
        buf162 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_201, buf161, reinterpret_tensor(primals_200, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf162)
        del primals_201
        buf163 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_31, out_9, x_10], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf163, buf150, primals_197, buf162, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf165 = buf155; del buf155  # reuse
        buf166 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [input_85, input_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf166, buf164, primals_203, primals_204, primals_205, primals_206, primals_207, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_10], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf167, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_10], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf166, primals_209, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf168, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf169 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf170 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_32, joi_feat_33], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf167, buf168, primals_210, primals_211, primals_212, primals_213, buf169, buf170, 32768, grid=grid(32768), stream=stream0)
        del buf167
        buf171 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf172 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_34, adaptive_avg_pool2d_10], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf172, buf170, primals_214, 512, 64, grid=grid(512), stream=stream0)
        buf173 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf172, (4, 128), (128, 1), 0), reinterpret_tensor(primals_215, (128, 8), (1, 128), 0), out=buf173)
        buf174 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [input_87, input_88], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf174, primals_216, 32, grid=grid(32), stream=stream0)
        del primals_216
        buf175 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_218, buf174, reinterpret_tensor(primals_217, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf175)
        del primals_218
        buf176 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_34, out_10, x_11], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf176, buf163, primals_214, buf175, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf178 = buf168; del buf168  # reuse
        buf179 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf179, buf177, primals_220, primals_221, primals_222, primals_223, primals_224, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_11], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_225, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf180, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_11], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf179, primals_226, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf181, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf182 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf183 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_35, joi_feat_36], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf180, buf181, primals_227, primals_228, primals_229, primals_230, buf182, buf183, 32768, grid=grid(32768), stream=stream0)
        del buf180
        buf184 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf185 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_37, adaptive_avg_pool2d_11], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf185, buf183, primals_231, 512, 64, grid=grid(512), stream=stream0)
        buf186 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf185, (4, 128), (128, 1), 0), reinterpret_tensor(primals_232, (128, 8), (1, 128), 0), out=buf186)
        buf187 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [input_94, input_95], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf187, primals_233, 32, grid=grid(32), stream=stream0)
        del primals_233
        buf188 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_235, buf187, reinterpret_tensor(primals_234, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf188)
        del primals_235
        buf189 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_37, out_11, x_12], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf189, buf176, primals_231, buf188, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf191 = buf181; del buf181  # reuse
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [input_99, input_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf192, buf190, primals_237, primals_238, primals_239, primals_240, primals_241, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_12], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf193, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_12], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf192, primals_243, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf194, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf195 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf196 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_38, joi_feat_39], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf193, buf194, primals_244, primals_245, primals_246, primals_247, buf195, buf196, 32768, grid=grid(32768), stream=stream0)
        del buf193
        buf197 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf198 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_40, adaptive_avg_pool2d_12], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf198, buf196, primals_248, 512, 64, grid=grid(512), stream=stream0)
        buf199 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_101], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf198, (4, 128), (128, 1), 0), reinterpret_tensor(primals_249, (128, 8), (1, 128), 0), out=buf199)
        buf200 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [input_101, input_102], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf200, primals_250, 32, grid=grid(32), stream=stream0)
        del primals_250
        buf201 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_252, buf200, reinterpret_tensor(primals_251, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf201)
        del primals_252
        buf202 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_40, out_12, x_13], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf202, buf189, primals_248, buf201, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_105], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf204 = buf194; del buf194  # reuse
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [input_106, input_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf205, buf203, primals_254, primals_255, primals_256, primals_257, primals_258, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_13], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_259, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf206, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_13], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf205, primals_260, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf207, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf208 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf209 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_41, joi_feat_42], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf206, buf207, primals_261, primals_262, primals_263, primals_264, buf208, buf209, 32768, grid=grid(32768), stream=stream0)
        del buf206
        buf210 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf211 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_43, adaptive_avg_pool2d_13], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf211, buf209, primals_265, 512, 64, grid=grid(512), stream=stream0)
        buf212 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf211, (4, 128), (128, 1), 0), reinterpret_tensor(primals_266, (128, 8), (1, 128), 0), out=buf212)
        buf213 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [input_108, input_109], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf213, primals_267, 32, grid=grid(32), stream=stream0)
        del primals_267
        buf214 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_269, buf213, reinterpret_tensor(primals_268, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf214)
        del primals_269
        buf215 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_43, out_13, x_14], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf215, buf202, primals_265, buf214, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf217 = buf207; del buf207  # reuse
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [input_113, input_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf218, buf216, primals_271, primals_272, primals_273, primals_274, primals_275, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_14], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_276, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf219, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_14], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf218, primals_277, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf220, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf221 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf222 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_44, joi_feat_45], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf219, buf220, primals_278, primals_279, primals_280, primals_281, buf221, buf222, 32768, grid=grid(32768), stream=stream0)
        del buf219
        buf223 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf224 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_46, adaptive_avg_pool2d_14], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf224, buf222, primals_282, 512, 64, grid=grid(512), stream=stream0)
        buf225 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf224, (4, 128), (128, 1), 0), reinterpret_tensor(primals_283, (128, 8), (1, 128), 0), out=buf225)
        buf226 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [input_115, input_116], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf226, primals_284, 32, grid=grid(32), stream=stream0)
        del primals_284
        buf227 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_286, buf226, reinterpret_tensor(primals_285, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf227)
        del primals_286
        buf228 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_46, out_14, x_15], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf228, buf215, primals_282, buf227, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf230 = buf220; del buf220  # reuse
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [input_120, input_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf231, buf229, primals_288, primals_289, primals_290, primals_291, primals_292, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_15], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_293, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf232, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_15], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf231, primals_294, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf233, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf234 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf235 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_47, joi_feat_48], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf232, buf233, primals_295, primals_296, primals_297, primals_298, buf234, buf235, 32768, grid=grid(32768), stream=stream0)
        del buf232
        buf236 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf237 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_49, adaptive_avg_pool2d_15], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf237, buf235, primals_299, 512, 64, grid=grid(512), stream=stream0)
        buf238 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf237, (4, 128), (128, 1), 0), reinterpret_tensor(primals_300, (128, 8), (1, 128), 0), out=buf238)
        buf239 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [input_122, input_123], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf239, primals_301, 32, grid=grid(32), stream=stream0)
        del primals_301
        buf240 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_303, buf239, reinterpret_tensor(primals_302, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf240)
        del primals_303
        buf241 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_49, out_15, x_16], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf241, buf228, primals_299, buf240, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_126], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf243 = buf233; del buf233  # reuse
        buf244 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [input_127, input_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf244, buf242, primals_305, primals_306, primals_307, primals_308, primals_309, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_16], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_310, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf245, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_16], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf244, primals_311, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf246, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf247 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf248 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_50, joi_feat_51], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf245, buf246, primals_312, primals_313, primals_314, primals_315, buf247, buf248, 32768, grid=grid(32768), stream=stream0)
        del buf245
        buf249 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf250 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_52, adaptive_avg_pool2d_16], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf250, buf248, primals_316, 512, 64, grid=grid(512), stream=stream0)
        buf251 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_129], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf250, (4, 128), (128, 1), 0), reinterpret_tensor(primals_317, (128, 8), (1, 128), 0), out=buf251)
        buf252 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [input_129, input_130], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf252, primals_318, 32, grid=grid(32), stream=stream0)
        del primals_318
        buf253 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_320, buf252, reinterpret_tensor(primals_319, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf253)
        del primals_320
        buf254 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_52, out_16, x_17], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf254, buf241, primals_316, buf253, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_133], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf256 = buf246; del buf246  # reuse
        buf257 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [input_134, input_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf257, buf255, primals_322, primals_323, primals_324, primals_325, primals_326, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_17], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf258, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_17], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf257, primals_328, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf259, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf260 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf261 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_53, joi_feat_54], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf258, buf259, primals_329, primals_330, primals_331, primals_332, buf260, buf261, 32768, grid=grid(32768), stream=stream0)
        del buf258
        buf262 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf263 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_55, adaptive_avg_pool2d_17], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf263, buf261, primals_333, 512, 64, grid=grid(512), stream=stream0)
        buf264 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf263, (4, 128), (128, 1), 0), reinterpret_tensor(primals_334, (128, 8), (1, 128), 0), out=buf264)
        buf265 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [input_136, input_137], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf265, primals_335, 32, grid=grid(32), stream=stream0)
        del primals_335
        buf266 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_138], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_337, buf265, reinterpret_tensor(primals_336, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf266)
        del primals_337
        buf267 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_55, out_17, x_18], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf267, buf254, primals_333, buf266, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_140], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_338, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf269 = buf259; del buf259  # reuse
        buf270 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [input_141, input_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf270, buf268, primals_339, primals_340, primals_341, primals_342, primals_343, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_18], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_344, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf271, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_18], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf270, primals_345, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf272, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf273 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf274 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_56, joi_feat_57], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf271, buf272, primals_346, primals_347, primals_348, primals_349, buf273, buf274, 32768, grid=grid(32768), stream=stream0)
        del buf271
        buf275 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_58, adaptive_avg_pool2d_18], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf276, buf274, primals_350, 512, 64, grid=grid(512), stream=stream0)
        buf277 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_143], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf276, (4, 128), (128, 1), 0), reinterpret_tensor(primals_351, (128, 8), (1, 128), 0), out=buf277)
        buf278 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [input_143, input_144], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf278, primals_352, 32, grid=grid(32), stream=stream0)
        del primals_352
        buf279 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_354, buf278, reinterpret_tensor(primals_353, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf279)
        del primals_354
        buf280 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_58, out_18, x_19], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf280, buf267, primals_350, buf279, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_355, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf282 = buf272; del buf272  # reuse
        buf283 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [input_148, input_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf283, buf281, primals_356, primals_357, primals_358, primals_359, primals_360, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_19], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_361, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf284, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_19], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf283, primals_362, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf285, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf286 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf287 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_59, joi_feat_60], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf284, buf285, primals_363, primals_364, primals_365, primals_366, buf286, buf287, 32768, grid=grid(32768), stream=stream0)
        del buf284
        buf288 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf289 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_61, adaptive_avg_pool2d_19], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf289, buf287, primals_367, 512, 64, grid=grid(512), stream=stream0)
        buf290 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_150], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf289, (4, 128), (128, 1), 0), reinterpret_tensor(primals_368, (128, 8), (1, 128), 0), out=buf290)
        buf291 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [input_150, input_151], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf291, primals_369, 32, grid=grid(32), stream=stream0)
        del primals_369
        buf292 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_152], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_371, buf291, reinterpret_tensor(primals_370, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf292)
        del primals_371
        buf293 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_61, out_19, x_20], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf293, buf280, primals_367, buf292, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf295 = buf285; del buf285  # reuse
        buf296 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [input_155, input_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf296, buf294, primals_373, primals_374, primals_375, primals_376, primals_377, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_20], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, primals_378, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf297, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_20], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf296, primals_379, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf298, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf299 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf300 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_62, joi_feat_63], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf297, buf298, primals_380, primals_381, primals_382, primals_383, buf299, buf300, 32768, grid=grid(32768), stream=stream0)
        del buf297
        buf301 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf302 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_64, adaptive_avg_pool2d_20], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf302, buf300, primals_384, 512, 64, grid=grid(512), stream=stream0)
        buf303 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_157], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf302, (4, 128), (128, 1), 0), reinterpret_tensor(primals_385, (128, 8), (1, 128), 0), out=buf303)
        buf304 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [input_157, input_158], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf304, primals_386, 32, grid=grid(32), stream=stream0)
        del primals_386
        buf305 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_388, buf304, reinterpret_tensor(primals_387, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf305)
        del primals_388
        buf306 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_64, out_20, x_21], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf306, buf293, primals_384, buf305, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_161], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, primals_389, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf308 = buf298; del buf298  # reuse
        buf309 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [input_162, input_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf309, buf307, primals_390, primals_391, primals_392, primals_393, primals_394, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_21], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_395, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf310, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_21], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf309, primals_396, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf311, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf312 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf313 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_65, joi_feat_66], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf310, buf311, primals_397, primals_398, primals_399, primals_400, buf312, buf313, 32768, grid=grid(32768), stream=stream0)
        del buf310
        buf314 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf315 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_67, adaptive_avg_pool2d_21], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf315, buf313, primals_401, 512, 64, grid=grid(512), stream=stream0)
        buf316 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_164], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf315, (4, 128), (128, 1), 0), reinterpret_tensor(primals_402, (128, 8), (1, 128), 0), out=buf316)
        buf317 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [input_164, input_165], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf317, primals_403, 32, grid=grid(32), stream=stream0)
        del primals_403
        buf318 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_405, buf317, reinterpret_tensor(primals_404, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf318)
        del primals_405
        buf319 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_67, out_21, x_22], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf319, buf306, primals_401, buf318, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_406, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf321 = buf311; del buf311  # reuse
        buf322 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [input_169, input_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf322, buf320, primals_407, primals_408, primals_409, primals_410, primals_411, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_22], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf323, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_22], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf322, primals_413, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf324, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf325 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf326 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_68, joi_feat_69], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf323, buf324, primals_414, primals_415, primals_416, primals_417, buf325, buf326, 32768, grid=grid(32768), stream=stream0)
        del buf323
        buf327 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf328 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_70, adaptive_avg_pool2d_22], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf328, buf326, primals_418, 512, 64, grid=grid(512), stream=stream0)
        buf329 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf328, (4, 128), (128, 1), 0), reinterpret_tensor(primals_419, (128, 8), (1, 128), 0), out=buf329)
        buf330 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [input_171, input_172], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf330, primals_420, 32, grid=grid(32), stream=stream0)
        del primals_420
        buf331 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_173], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_422, buf330, reinterpret_tensor(primals_421, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf331)
        del primals_422
        buf332 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_70, out_22, x_23], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_31.run(buf332, buf319, primals_418, buf331, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_175], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_423, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf334 = buf324; del buf324  # reuse
        buf335 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [input_176, input_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_28.run(buf335, buf333, primals_424, primals_425, primals_426, primals_427, primals_428, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [loc_23], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_429, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf336, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [sur_23], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf335, primals_430, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf337, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf338 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf339 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_71, joi_feat_72], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_29.run(buf336, buf337, primals_431, primals_432, primals_433, primals_434, buf338, buf339, 32768, grid=grid(32768), stream=stream0)
        del buf336
        del buf337
        buf340 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf341 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_73, adaptive_avg_pool2d_23], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_30.run(buf341, buf339, primals_435, 512, 64, grid=grid(512), stream=stream0)
        buf342 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_178], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf341, (4, 128), (128, 1), 0), reinterpret_tensor(primals_436, (128, 8), (1, 128), 0), out=buf342)
        buf343 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [input_178, input_179], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_15.run(buf343, primals_437, 32, grid=grid(32), stream=stream0)
        del primals_437
        buf344 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_439, buf343, reinterpret_tensor(primals_438, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf344)
        del primals_439
        buf345 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf347 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_26, input_182, input_183], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_32.run(buf85, buf332, buf339, primals_435, buf344, primals_440, primals_441, primals_442, primals_443, primals_444, buf345, buf347, 256, 256, grid=grid(256, 256), stream=stream0)
        del buf339
    return (buf20, buf69, buf347, buf0, buf1, primals_3, primals_4, primals_5, primals_6, primals_7, buf2, primals_9, primals_10, primals_11, primals_12, primals_13, buf3, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, buf4, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_77, primals_78, primals_79, primals_80, primals_81, buf5, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_440, primals_441, primals_442, primals_443, primals_444, buf8, buf10, buf11, buf13, buf14, buf18, buf20, buf22, buf24, buf27, buf29, buf30, reinterpret_tensor(buf33, (4, 64), (64, 1), 0), buf35, buf36, buf37, buf38, buf40, buf43, reinterpret_tensor(buf47, (4, 64), (64, 1), 0), buf49, buf50, buf51, buf52, buf54, buf57, reinterpret_tensor(buf61, (4, 64), (64, 1), 0), buf63, buf64, buf67, buf69, buf71, buf73, buf76, buf78, buf79, reinterpret_tensor(buf81, (4, 128), (128, 1), 0), buf83, buf84, buf85, buf86, buf88, buf91, reinterpret_tensor(buf94, (4, 128), (128, 1), 0), buf96, buf97, buf98, buf99, buf101, buf104, reinterpret_tensor(buf107, (4, 128), (128, 1), 0), buf109, buf110, buf111, buf112, buf114, buf117, reinterpret_tensor(buf120, (4, 128), (128, 1), 0), buf122, buf123, buf124, buf125, buf127, buf130, reinterpret_tensor(buf133, (4, 128), (128, 1), 0), buf135, buf136, buf137, buf138, buf140, buf143, reinterpret_tensor(buf146, (4, 128), (128, 1), 0), buf148, buf149, buf150, buf151, buf153, buf156, reinterpret_tensor(buf159, (4, 128), (128, 1), 0), buf161, buf162, buf163, buf164, buf166, buf169, reinterpret_tensor(buf172, (4, 128), (128, 1), 0), buf174, buf175, buf176, buf177, buf179, buf182, reinterpret_tensor(buf185, (4, 128), (128, 1), 0), buf187, buf188, buf189, buf190, buf192, buf195, reinterpret_tensor(buf198, (4, 128), (128, 1), 0), buf200, buf201, buf202, buf203, buf205, buf208, reinterpret_tensor(buf211, (4, 128), (128, 1), 0), buf213, buf214, buf215, buf216, buf218, buf221, reinterpret_tensor(buf224, (4, 128), (128, 1), 0), buf226, buf227, buf228, buf229, buf231, buf234, reinterpret_tensor(buf237, (4, 128), (128, 1), 0), buf239, buf240, buf241, buf242, buf244, buf247, reinterpret_tensor(buf250, (4, 128), (128, 1), 0), buf252, buf253, buf254, buf255, buf257, buf260, reinterpret_tensor(buf263, (4, 128), (128, 1), 0), buf265, buf266, buf267, buf268, buf270, buf273, reinterpret_tensor(buf276, (4, 128), (128, 1), 0), buf278, buf279, buf280, buf281, buf283, buf286, reinterpret_tensor(buf289, (4, 128), (128, 1), 0), buf291, buf292, buf293, buf294, buf296, buf299, reinterpret_tensor(buf302, (4, 128), (128, 1), 0), buf304, buf305, buf306, buf307, buf309, buf312, reinterpret_tensor(buf315, (4, 128), (128, 1), 0), buf317, buf318, buf319, buf320, buf322, buf325, reinterpret_tensor(buf328, (4, 128), (128, 1), 0), buf330, buf331, buf332, buf333, buf335, buf338, reinterpret_tensor(buf341, (4, 128), (128, 1), 0), buf343, buf344, buf345, primals_438, primals_436, primals_421, primals_419, primals_404, primals_402, primals_387, primals_385, primals_370, primals_368, primals_353, primals_351, primals_336, primals_334, primals_319, primals_317, primals_302, primals_300, primals_285, primals_283, primals_268, primals_266, primals_251, primals_249, primals_234, primals_232, primals_217, primals_215, primals_200, primals_198, primals_183, primals_181, primals_166, primals_164, primals_149, primals_147, primals_132, primals_130, primals_115, primals_113, primals_98, primals_96, primals_75, primals_73, primals_58, primals_56, primals_41, primals_39, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((35, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((35, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((35, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((35, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((35, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 35, 3, 3), (315, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((8, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((8, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((8, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((131, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((131, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((131, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((131, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((131, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 131, 3, 3), (1179, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
