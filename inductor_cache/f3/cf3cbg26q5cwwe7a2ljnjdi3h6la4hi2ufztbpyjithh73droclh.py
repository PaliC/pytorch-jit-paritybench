# AOT ID: ['21_forward']
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


# kernel path: inductor_cache/3h/c3hpecenggnrxcqg3nyshgwbawgb3vfvl2axvlomnkql4xptnmer.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/nu/cnuedntizcvgrf3a7y7wp5fpvjeo2fjv3uant6ubkp6mi5baj43r.py
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
    ynumel = 2560
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


# kernel path: inductor_cache/3t/c3tbywwh3xnx3xkd3xlhufyyj6q36pztxqs67cyklgfjdf3texpy.py
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
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jn/cjn5w7gsvplyg3zxajhwykbdxbl2fjadrukprnunafhiju4vdw4n.py
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
    ynumel = 51200
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/24/c242whermvz3yz5cx4kyjucecqze6q2t3pnvh7s6d3uuso2qntns.py
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
    ynumel = 102400
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


# kernel path: inductor_cache/yk/cykavycigjmp5khkmwcgwkvxr7mrvx5ulrhhgtehnai5yjsdjvyc.py
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
    ynumel = 204800
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


# kernel path: inductor_cache/6u/c6ufmo5fw5yvvkbx5d4xbyoyxll7rp6xqzzlix575pgpl673rel6.py
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
    ynumel = 409600
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 640)
    y1 = yindex // 640
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 640*x2 + 5760*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ie/ciexeegzt7lq3ige4u2a5gigqi26eyrvnqyrx3tvzfp2vh3t3ido.py
# Topologically Sorted Source Nodes: [sub, x], Original ATen: [aten.sub, aten.div]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
triton_poi_fused_div_sub_7 = async_compile.triton('triton_poi_fused_div_sub_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sub_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_sub_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/73/c73l3hd5omd3fjtpewblbxrxhzqqxrolzefq2tsr6tmdsc2mq2np.py
# Topologically Sorted Source Nodes: [batch_norm, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub_1
#   x_1 => relu
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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


# kernel path: inductor_cache/5d/c5dw7ialxym4zwgo6md3gynax7hcs7yczlhbxyl3mpcg72oftitp.py
# Topologically Sorted Source Nodes: [batch_norm_1, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_4, mul_5, sub_2
#   out_1 => relu_1
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
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


# kernel path: inductor_cache/gy/cgy4g5mpbybi63tidtwuobbutlhzgyeskx2ekgq4rnveszb4wup2.py
# Topologically Sorted Source Nodes: [input_1, batch_norm_2, out_3], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_2 => add_6, mul_7, mul_8, sub_3
#   input_1 => add_4
#   out_3 => relu_2
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/rx/crxpdkf5finplavap3ronha22snexkfafwarngyv6elan26ynum7.py
# Topologically Sorted Source Nodes: [input_1, input_2, batch_norm_4, out_6], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_4 => add_11, mul_13, mul_14, sub_5
#   input_1 => add_4
#   input_2 => add_9
#   out_6 => relu_4
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_2), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_5), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %unsqueeze_442), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %unsqueeze_466), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tmp23 = tmp2 - tmp22
    tl.store(out_ptr0 + (x2), tmp21, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
    tl.store(out_ptr2 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/vu/cvup4zgkjq3jhhprux2yxz5oueo2wv4otgxs7rbbh72kfmydf3i2.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, batch_norm_6, out_9], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_6 => add_16, mul_19, mul_20, sub_7
#   input_1 => add_4
#   input_2 => add_9
#   input_3 => add_14
#   out_9 => relu_6
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_2), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_5), kwargs = {})
#   %add_14 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convolution_7), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_14, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_14, %unsqueeze_418), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x2), tmp23, None)
    tl.store(out_ptr1 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/62/c62socnrndhqbajraxhzsx7rfxqbbs5673hnooc3yukni6cu7jgs.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, batch_norm_8, x_2], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_8 => add_21, mul_25, mul_26, sub_9
#   input_1 => add_4
#   input_2 => add_9
#   input_3 => add_14
#   input_4 => add_19
#   x_2 => relu_8
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_2), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_5), kwargs = {})
#   %add_14 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convolution_7), kwargs = {})
#   %add_19 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %convolution_9), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %unsqueeze_394), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 160)
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
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
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(out_ptr0 + (x0), tmp25, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/kz/ckzqsgmwxr26noqtn2vj4uy3iftwjyd6h2hzrolzroxxtz2mcpez.py
# Topologically Sorted Source Nodes: [batch_norm_9, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_9 => add_23, mul_28, mul_29, sub_10
#   out_12 => relu_9
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
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


# kernel path: inductor_cache/je/cje45ofzmjsbrupxeyadg4d32vnrrc6tkz4d7bmlq2ggbseuoj3e.py
# Topologically Sorted Source Nodes: [input_5, batch_norm_10, out_14], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_10 => add_26, mul_31, mul_32, sub_11
#   input_5 => add_24
#   out_14 => relu_10
# Graph fragment:
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %unsqueeze_370), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/fd/cfd5yabqqqpsetvkykqy5p53chgc7x3gsa4t5qugpfxvw3dhqoku.py
# Topologically Sorted Source Nodes: [input_5, input_6, batch_norm_12, out_17], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_12 => add_31, mul_37, mul_38, sub_13
#   input_5 => add_24
#   input_6 => add_29
#   out_17 => relu_12
# Graph fragment:
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %convolution_14), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_29, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_29, %unsqueeze_346), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x2), tmp21, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/5j/c5j23x6a25u4hbwuqhvz5jic3od3thdeiuoz3d3gint62coat5fq.py
# Topologically Sorted Source Nodes: [input_5, input_6, input_7, batch_norm_14, out_20], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_14 => add_36, mul_43, mul_44, sub_15
#   input_5 => add_24
#   input_6 => add_29
#   input_7 => add_34
#   out_20 => relu_14
# Graph fragment:
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %convolution_14), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %convolution_16), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_36,), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %unsqueeze_322), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x2), tmp23, None)
    tl.store(out_ptr1 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/pw/cpwhgbwytt7grwtcpahkghf6ochdcfkwihm7agxzrl5ygt3j5o73.py
# Topologically Sorted Source Nodes: [input_5, input_6, input_7, input_8, batch_norm_16, x_3], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_16 => add_41, mul_49, mul_50, sub_17
#   input_5 => add_24
#   input_6 => add_29
#   input_7 => add_34
#   input_8 => add_39
#   x_3 => relu_16
# Graph fragment:
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %convolution_14), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %convolution_16), kwargs = {})
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %convolution_18), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_39, %unsqueeze_129), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_131), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_39, %unsqueeze_298), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
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
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(out_ptr0 + (x0), tmp25, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/v5/cv5ttcxaugeylfnzjg3ikrfkwprjxicokmofuqdgdxyqcqe3qpgu.py
# Topologically Sorted Source Nodes: [batch_norm_17, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_17 => add_43, mul_52, mul_53, sub_18
#   out_23 => relu_17
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_137), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_139), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %relu_17 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_43,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
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


# kernel path: inductor_cache/7j/c7ju4gmpe5etd27rpqf2lzrxeslmft6sqkpgirlxjee5ps3qhrdo.py
# Topologically Sorted Source Nodes: [input_9, batch_norm_18, out_25], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_18 => add_46, mul_55, mul_56, sub_19
#   input_9 => add_44
#   out_25 => relu_18
# Graph fragment:
#   %add_44 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %convolution_20), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_44, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %relu_18 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_46,), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_44, %unsqueeze_274), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/ti/ctikbtlldwyfq4vdr7e6cwrbcm35hd4mnfhu5e2fhjkkd25le52o.py
# Topologically Sorted Source Nodes: [input_9, input_10, batch_norm_20, out_28], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_20 => add_51, mul_61, mul_62, sub_21
#   input_10 => add_49
#   input_9 => add_44
#   out_28 => relu_20
# Graph fragment:
#   %add_44 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %convolution_20), kwargs = {})
#   %add_49 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %convolution_23), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_49, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %relu_20 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_51,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_49, %unsqueeze_250), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x2), tmp21, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/kc/ckccyewrooogdnlmurtrkxxnrw7ea6oqwdzdtd7umwrtxdy7ezpe.py
# Topologically Sorted Source Nodes: [input_9, input_10, input_11, batch_norm_22, out_31], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_22 => add_56, mul_67, mul_68, sub_23
#   input_10 => add_49
#   input_11 => add_54
#   input_9 => add_44
#   out_31 => relu_22
# Graph fragment:
#   %add_44 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %convolution_20), kwargs = {})
#   %add_49 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %convolution_23), kwargs = {})
#   %add_54 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %convolution_25), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_54, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_54, %unsqueeze_226), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x2), tmp23, None)
    tl.store(out_ptr1 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/zo/czognqtgxppsccyskeuecnv672xo64spm64nplcsylso7llwv22s.py
# Topologically Sorted Source Nodes: [input_9, input_10, input_11, input_12, batch_norm_24, out_34], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_24 => add_61, mul_73, mul_74, sub_25
#   input_10 => add_49
#   input_11 => add_54
#   input_12 => add_59
#   input_9 => add_44
#   out_34 => relu_24
# Graph fragment:
#   %add_44 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %convolution_20), kwargs = {})
#   %add_49 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %convolution_23), kwargs = {})
#   %add_54 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %convolution_25), kwargs = {})
#   %add_59 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %convolution_27), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_59, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_61,), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_59, %unsqueeze_202), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 640)
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
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
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(out_ptr0 + (x0), tmp25, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/zq/czqu2sauwyszoerskyf63oahxoyhmeyrfluwzngclz33mukj7lrg.py
# Topologically Sorted Source Nodes: [out_36], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   out_36 => view
# Graph fragment:
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%avg_pool2d, [-1, 640]), kwargs = {})
triton_poi_fused_view_24 = async_compile.triton('triton_poi_fused_view_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 640)
    x1 = xindex // 640
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (640*((x0 % 4)) + 2560*((x0 + 640*x1) // 2560) + ((((x0 + 640*x1) // 4) % 640))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_4, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (160, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_10, (160, ), (1, ))
    assert_size_stride(primals_11, (160, ), (1, ))
    assert_size_stride(primals_12, (160, ), (1, ))
    assert_size_stride(primals_13, (160, ), (1, ))
    assert_size_stride(primals_14, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_15, (160, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_16, (160, ), (1, ))
    assert_size_stride(primals_17, (160, ), (1, ))
    assert_size_stride(primals_18, (160, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_20, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_22, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_24, (160, ), (1, ))
    assert_size_stride(primals_25, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_26, (160, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_28, (160, ), (1, ))
    assert_size_stride(primals_29, (160, ), (1, ))
    assert_size_stride(primals_30, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_31, (160, ), (1, ))
    assert_size_stride(primals_32, (160, ), (1, ))
    assert_size_stride(primals_33, (160, ), (1, ))
    assert_size_stride(primals_34, (160, ), (1, ))
    assert_size_stride(primals_35, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_36, (160, ), (1, ))
    assert_size_stride(primals_37, (160, ), (1, ))
    assert_size_stride(primals_38, (160, ), (1, ))
    assert_size_stride(primals_39, (160, ), (1, ))
    assert_size_stride(primals_40, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_41, (160, ), (1, ))
    assert_size_stride(primals_42, (160, ), (1, ))
    assert_size_stride(primals_43, (160, ), (1, ))
    assert_size_stride(primals_44, (160, ), (1, ))
    assert_size_stride(primals_45, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_46, (160, ), (1, ))
    assert_size_stride(primals_47, (160, ), (1, ))
    assert_size_stride(primals_48, (160, ), (1, ))
    assert_size_stride(primals_49, (160, ), (1, ))
    assert_size_stride(primals_50, (320, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_51, (320, ), (1, ))
    assert_size_stride(primals_52, (320, ), (1, ))
    assert_size_stride(primals_53, (320, ), (1, ))
    assert_size_stride(primals_54, (320, ), (1, ))
    assert_size_stride(primals_55, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_56, (320, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_57, (320, ), (1, ))
    assert_size_stride(primals_58, (320, ), (1, ))
    assert_size_stride(primals_59, (320, ), (1, ))
    assert_size_stride(primals_60, (320, ), (1, ))
    assert_size_stride(primals_61, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_62, (320, ), (1, ))
    assert_size_stride(primals_63, (320, ), (1, ))
    assert_size_stride(primals_64, (320, ), (1, ))
    assert_size_stride(primals_65, (320, ), (1, ))
    assert_size_stride(primals_66, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_67, (320, ), (1, ))
    assert_size_stride(primals_68, (320, ), (1, ))
    assert_size_stride(primals_69, (320, ), (1, ))
    assert_size_stride(primals_70, (320, ), (1, ))
    assert_size_stride(primals_71, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_72, (320, ), (1, ))
    assert_size_stride(primals_73, (320, ), (1, ))
    assert_size_stride(primals_74, (320, ), (1, ))
    assert_size_stride(primals_75, (320, ), (1, ))
    assert_size_stride(primals_76, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_77, (320, ), (1, ))
    assert_size_stride(primals_78, (320, ), (1, ))
    assert_size_stride(primals_79, (320, ), (1, ))
    assert_size_stride(primals_80, (320, ), (1, ))
    assert_size_stride(primals_81, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_82, (320, ), (1, ))
    assert_size_stride(primals_83, (320, ), (1, ))
    assert_size_stride(primals_84, (320, ), (1, ))
    assert_size_stride(primals_85, (320, ), (1, ))
    assert_size_stride(primals_86, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_87, (320, ), (1, ))
    assert_size_stride(primals_88, (320, ), (1, ))
    assert_size_stride(primals_89, (320, ), (1, ))
    assert_size_stride(primals_90, (320, ), (1, ))
    assert_size_stride(primals_91, (640, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_92, (640, ), (1, ))
    assert_size_stride(primals_93, (640, ), (1, ))
    assert_size_stride(primals_94, (640, ), (1, ))
    assert_size_stride(primals_95, (640, ), (1, ))
    assert_size_stride(primals_96, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_97, (640, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_98, (640, ), (1, ))
    assert_size_stride(primals_99, (640, ), (1, ))
    assert_size_stride(primals_100, (640, ), (1, ))
    assert_size_stride(primals_101, (640, ), (1, ))
    assert_size_stride(primals_102, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_103, (640, ), (1, ))
    assert_size_stride(primals_104, (640, ), (1, ))
    assert_size_stride(primals_105, (640, ), (1, ))
    assert_size_stride(primals_106, (640, ), (1, ))
    assert_size_stride(primals_107, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_108, (640, ), (1, ))
    assert_size_stride(primals_109, (640, ), (1, ))
    assert_size_stride(primals_110, (640, ), (1, ))
    assert_size_stride(primals_111, (640, ), (1, ))
    assert_size_stride(primals_112, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_113, (640, ), (1, ))
    assert_size_stride(primals_114, (640, ), (1, ))
    assert_size_stride(primals_115, (640, ), (1, ))
    assert_size_stride(primals_116, (640, ), (1, ))
    assert_size_stride(primals_117, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_118, (640, ), (1, ))
    assert_size_stride(primals_119, (640, ), (1, ))
    assert_size_stride(primals_120, (640, ), (1, ))
    assert_size_stride(primals_121, (640, ), (1, ))
    assert_size_stride(primals_122, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_123, (640, ), (1, ))
    assert_size_stride(primals_124, (640, ), (1, ))
    assert_size_stride(primals_125, (640, ), (1, ))
    assert_size_stride(primals_126, (640, ), (1, ))
    assert_size_stride(primals_127, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_128, (640, ), (1, ))
    assert_size_stride(primals_129, (640, ), (1, ))
    assert_size_stride(primals_130, (640, ), (1, ))
    assert_size_stride(primals_131, (640, ), (1, ))
    assert_size_stride(primals_132, (10, 640), (640, 1))
    assert_size_stride(primals_133, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_4, buf0, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_4
        buf1 = empty_strided_cuda((160, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_9, buf1, 2560, 9, grid=grid(2560, 9), stream=stream0)
        del primals_9
        buf2 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_14, buf2, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_14
        buf3 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_20, buf3, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_20
        buf4 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_25, buf4, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_25
        buf5 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_30, buf5, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_30
        buf6 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_35, buf6, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_35
        buf7 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_40, buf7, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_40
        buf8 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_45, buf8, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_45
        buf9 = empty_strided_cuda((320, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_50, buf9, 51200, 9, grid=grid(51200, 9), stream=stream0)
        del primals_50
        buf10 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_55, buf10, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_55
        buf11 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_61, buf11, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_61
        buf12 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_66, buf12, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_66
        buf13 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_71, buf13, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_71
        buf14 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_76, buf14, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_76
        buf15 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_81, buf15, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_81
        buf16 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_86, buf16, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_86
        buf17 = empty_strided_cuda((640, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_91, buf17, 204800, 9, grid=grid(204800, 9), stream=stream0)
        del primals_91
        buf18 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_96, buf18, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_96
        buf19 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_102, buf19, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_102
        buf20 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_107, buf20, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_107
        buf21 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_112, buf21, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_112
        buf22 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_117, buf22, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_117
        buf23 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_122, buf23, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_122
        buf24 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_127, buf24, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_127
        buf25 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x], Original ATen: [aten.sub, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sub_7.run(primals_2, primals_1, primals_3, buf25, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 16, 64, 64), (65536, 1, 1024, 16))
        buf27 = empty_strided_cuda((4, 16, 64, 64), (65536, 1, 1024, 16), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf26, primals_5, primals_6, primals_7, primals_8, buf27, 262144, grid=grid(262144), stream=stream0)
        del primals_8
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf29 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_1, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf28, primals_10, primals_11, primals_12, primals_13, buf29, 2621440, grid=grid(2621440), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 160, 64, 64), (655360, 1, 10240, 160))
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf27, primals_15, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf32 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, batch_norm_2, out_3], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf31, buf30, primals_16, primals_17, primals_18, primals_19, buf32, 2621440, grid=grid(2621440), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf34 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_3, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf33, primals_21, primals_22, primals_23, primals_24, buf34, 2621440, grid=grid(2621440), stream=stream0)
        del primals_24
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf36 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        buf96 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        buf97 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, batch_norm_4, out_6], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_11.run(buf31, buf30, buf35, primals_26, primals_27, primals_28, primals_29, primals_16, buf36, buf96, buf97, 2621440, grid=grid(2621440), stream=stream0)
        del primals_16
        del primals_26
        del primals_29
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf38 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_5, out_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf37, primals_31, primals_32, primals_33, primals_34, buf38, 2621440, grid=grid(2621440), stream=stream0)
        del primals_34
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf40 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        buf95 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, batch_norm_6, out_9], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_12.run(buf31, buf30, buf35, buf39, primals_36, primals_37, primals_38, primals_39, buf40, buf95, 2621440, grid=grid(2621440), stream=stream0)
        del primals_36
        del primals_39
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf42 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_7, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf41, primals_41, primals_42, primals_43, primals_44, buf42, 2621440, grid=grid(2621440), stream=stream0)
        del primals_44
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf44 = buf31; del buf31  # reuse
        buf45 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        buf94 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, batch_norm_8, x_2], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13.run(buf44, buf30, buf35, buf39, buf43, primals_46, primals_47, primals_48, primals_49, buf45, buf94, 2621440, grid=grid(2621440), stream=stream0)
        del buf30
        del buf35
        del buf39
        del buf43
        del buf44
        del primals_46
        del primals_49
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf47 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_9, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf46, primals_51, primals_52, primals_53, primals_54, buf47, 1310720, grid=grid(1310720), stream=stream0)
        del primals_54
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 320, 32, 32), (327680, 1, 10240, 320))
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf45, primals_56, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf50 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        buf93 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, batch_norm_10, out_14], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15.run(buf49, buf48, primals_57, primals_58, primals_59, primals_60, buf50, buf93, 1310720, grid=grid(1310720), stream=stream0)
        del primals_57
        del primals_60
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf52 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_11, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf51, primals_62, primals_63, primals_64, primals_65, buf52, 1310720, grid=grid(1310720), stream=stream0)
        del primals_65
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf54 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        buf92 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, batch_norm_12, out_17], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_16.run(buf49, buf48, buf53, primals_67, primals_68, primals_69, primals_70, buf54, buf92, 1310720, grid=grid(1310720), stream=stream0)
        del primals_67
        del primals_70
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf56 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_13, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf55, primals_72, primals_73, primals_74, primals_75, buf56, 1310720, grid=grid(1310720), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [out_19], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf58 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        buf91 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, input_7, batch_norm_14, out_20], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17.run(buf49, buf48, buf53, buf57, primals_77, primals_78, primals_79, primals_80, buf58, buf91, 1310720, grid=grid(1310720), stream=stream0)
        del primals_77
        del primals_80
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf60 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_15, out_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf59, primals_82, primals_83, primals_84, primals_85, buf60, 1310720, grid=grid(1310720), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf62 = buf49; del buf49  # reuse
        buf63 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        buf90 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, input_7, input_8, batch_norm_16, x_3], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18.run(buf62, buf48, buf53, buf57, buf61, primals_87, primals_88, primals_89, primals_90, buf63, buf90, 1310720, grid=grid(1310720), stream=stream0)
        del buf48
        del buf53
        del buf57
        del buf61
        del buf62
        del primals_87
        del primals_90
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, buf17, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf65 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_17, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf64, primals_92, primals_93, primals_94, primals_95, buf65, 655360, grid=grid(655360), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 640, 16, 16), (163840, 1, 10240, 640))
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf63, primals_97, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf68 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        buf89 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, batch_norm_18, out_25], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20.run(buf67, buf66, primals_98, primals_99, primals_100, primals_101, buf68, buf89, 655360, grid=grid(655360), stream=stream0)
        del primals_101
        del primals_98
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf70 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_19, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf69, primals_103, primals_104, primals_105, primals_106, buf70, 655360, grid=grid(655360), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [out_27], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf72 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        buf88 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, input_10, batch_norm_20, out_28], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_21.run(buf67, buf66, buf71, primals_108, primals_109, primals_110, primals_111, buf72, buf88, 655360, grid=grid(655360), stream=stream0)
        del primals_108
        del primals_111
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf74 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_21, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf73, primals_113, primals_114, primals_115, primals_116, buf74, 655360, grid=grid(655360), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf76 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        buf87 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, input_10, input_11, batch_norm_22, out_31], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22.run(buf67, buf66, buf71, buf75, primals_118, primals_119, primals_120, primals_121, buf76, buf87, 655360, grid=grid(655360), stream=stream0)
        del primals_118
        del primals_121
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf78 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_23, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf77, primals_123, primals_124, primals_125, primals_126, buf78, 655360, grid=grid(655360), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf80 = buf67; del buf67  # reuse
        buf81 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        buf86 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, input_10, input_11, input_12, batch_norm_24, out_34], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23.run(buf80, buf66, buf71, buf75, buf79, primals_128, primals_129, primals_130, primals_131, buf81, buf86, 655360, grid=grid(655360), stream=stream0)
        del buf66
        del buf71
        del buf75
        del buf79
        del buf80
        del primals_128
        del primals_131
        # Topologically Sorted Source Nodes: [out_35], Original ATen: [aten.avg_pool2d]
        buf82 = torch.ops.aten.avg_pool2d.default(buf81, [8, 8], [8, 8], [0, 0], False, True, None)
        buf83 = buf82
        del buf82
        buf84 = empty_strided_cuda((16, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_36], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_24.run(buf83, buf84, 10240, grid=grid(10240), stream=stream0)
        del buf83
        buf85 = empty_strided_cuda((16, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_133, buf84, reinterpret_tensor(primals_132, (640, 10), (1, 640), 0), alpha=1, beta=1, out=buf85)
        del primals_133
    return (buf85, buf0, primals_5, primals_6, primals_7, buf1, primals_10, primals_11, primals_12, buf2, primals_15, primals_17, primals_18, buf3, primals_21, primals_22, primals_23, buf4, primals_27, primals_28, buf5, primals_31, primals_32, primals_33, buf6, primals_37, primals_38, buf7, primals_41, primals_42, primals_43, buf8, primals_47, primals_48, buf9, primals_51, primals_52, primals_53, buf10, primals_56, primals_58, primals_59, buf11, primals_62, primals_63, primals_64, buf12, primals_68, primals_69, buf13, primals_72, primals_73, primals_74, buf14, primals_78, primals_79, buf15, primals_82, primals_83, primals_84, buf16, primals_88, primals_89, buf17, primals_92, primals_93, primals_94, buf18, primals_97, primals_99, primals_100, buf19, primals_103, primals_104, primals_105, buf20, primals_109, primals_110, buf21, primals_113, primals_114, primals_115, buf22, primals_119, primals_120, buf23, primals_123, primals_124, primals_125, buf24, primals_129, primals_130, buf25, buf26, buf27, buf28, buf29, buf32, buf33, buf34, buf36, buf37, buf38, buf40, buf41, buf42, buf45, buf46, buf47, buf50, buf51, buf52, buf54, buf55, buf56, buf58, buf59, buf60, buf63, buf64, buf65, buf68, buf69, buf70, buf72, buf73, buf74, buf76, buf77, buf78, buf81, buf84, primals_132, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((160, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((160, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((320, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((320, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((640, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((640, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((10, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
