# AOT ID: ['53_forward']
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


# kernel path: inductor_cache/tq/ctqmv3m47kr5v5qfdnn3g63yo4feq5dt7gb4kn7icnsub4flk4jo.py
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
    size_hints={'y': 256, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 5
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
    tmp0 = tl.load(in_ptr0 + (x2 + 5*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 80*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/aj/cajggc75ojogldhn3325adbo3djb33tb6wf5xxdqaot4bmjgpdva.py
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
    size_hints={'y': 256, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/qq/cqqfqnvyk4zpe2do2wfob4qqnm5lxxcrlg67baqrrgxovc3f432l.py
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
    size_hints={'y': 256, 'x': 2}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 2
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
    tmp0 = tl.load(in_ptr0 + (x2 + 2*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 32*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xj/cxjarnnhoncfnd7oybfwa3tzajqxh5qrkurxaxercbtz4ingxlmr.py
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
    size_hints={'y': 256, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/cx/ccx745ahlvamnqyh5gilmjtkkuxl5arlpx5tfemqins2xig3nu2b.py
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
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/yn/cynshpyv4aedzikg6yfiqijudjk4x34i2k2tncuvp2jcni6xvhz7.py
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
    size_hints={'y': 1024, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 5
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 24)
    y1 = yindex // 24
    tmp0 = tl.load(in_ptr0 + (x2 + 5*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x2 + 120*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/bn/cbnftohctomakv6tzki3usta6plri263uyj4rynvn3cgeey66jd7.py
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
    size_hints={'y': 1024, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 24)
    y1 = yindex // 24
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x2 + 72*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ob/cobqovfmat66wxjlkqzjidzhsiaga5vwayaecgcexcaj7a3fhkct.py
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
    size_hints={'y': 1024, 'x': 2}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 2
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 24)
    y1 = yindex // 24
    tmp0 = tl.load(in_ptr0 + (x2 + 2*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x2 + 48*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zu/czufrxxvuyfvwdpdgnqw7b77jnrdqwp7pblpvami2zmgukxafxco.py
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
    size_hints={'y': 1024, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 24)
    y1 = yindex // 24
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x2 + 96*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/uj/cuj4tor5u7ekw5cve6jmhockb3z4hp4dxtvg5kbxmepokz7c4ly3.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 24)
    y1 = yindex // 24
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x2 + 216*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/wd/cwdsjhvcfmbgbi4b4znaydfurp67tezgvsxcx4ypd7igpla5mr5z.py
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
    size_hints={'y': 1024, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/b5/cb54fy7tpcwvhenzbkmdm5tou2iw4v6abtniysj7v7gc3vcko735.py
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
    size_hints={'y': 1024, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/z7/cz7c7zunlsfnmj7ki6exvf5oq5t44yyddrmrvu67o2u35fnsefza.py
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
    size_hints={'y': 1024, 'x': 2}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 2
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
    tmp0 = tl.load(in_ptr0 + (x2 + 2*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 64*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qz/cqzyzl62swm3zu5paluq2bfy6ctbjrcka6x52assjr5hrbnueoel.py
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
    size_hints={'y': 1024, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ci/cciqkyjhb65a4mp5c35nhplsfnwewyh2rjvqtb4fdoegtbnvchxb.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/4z/c4z3zazvq22x5ip4onkxqkufats7gq4tkoo5rd7rlkj2tb4ibuut.py
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
    size_hints={'y': 4096, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 5
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
    tmp0 = tl.load(in_ptr0 + (x2 + 5*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 320*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vl/cvldbnujp6wkcyxzapiiivdgou7bqyxepnwztgvc2mpl4xqxi5ec.py
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
    size_hints={'y': 4096, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/rj/crjm4egiomsg6svl226g4jdiegkpaglfzuhipqngtm55pnypuejz.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_19 = async_compile.triton('triton_poi_fused_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 2}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 2
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
    tmp0 = tl.load(in_ptr0 + (x2 + 2*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 128*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cr/ccrclwghsliv6h6m3ti2fewh7t5wr4a2w4aselm5rxymqkqjg5hs.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_20 = async_compile.triton('triton_poi_fused_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/on/con6u26cfcv4ysrtddm5pxunly2npemz5gv3w7rue5brdbuqnnzy.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_21 = async_compile.triton('triton_poi_fused_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/7b/c7bq2ttcllnskzd62has4vkip4y6tnnchcew7tgrps33zokixbwr.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => clamp_max, clamp_min
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0.0), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/ww/cwwbibyptj7sofafhbhx3stqyhhhmamihbkcipm6k2byra6bq5a5.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_4 => convolution_1
#   input_5 => add_3, mul_4, mul_5, sub_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=9] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/gf/cgflb7iurqbh7crs2fzq5zpjjxtuoorsqqxef2oyxnlpkuz2m2oa.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_6 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_3, %primals_14, %primals_15, [2, 2], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/4m/c4m7kumrleyt77wolwu442li5k4wb2icq5lxxpyktzlqfxycdoo3.py
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_18 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_3, %primals_38, %primals_39, [2, 2], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_25 = async_compile.triton('triton_poi_fused_convolution_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zh/czhcskesdifs3pc3mqhqhg6z7ibrjvrnjq4pgy2n3b3mkr4tkagz.py
# Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_24 => convolution_8
# Graph fragment:
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_3, %primals_50, %primals_51, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_26 = async_compile.triton('triton_poi_fused_convolution_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_26(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/me/cmecwueqtlorxmclxfyuervzxt7cnd6wiob6ebq3du3jpzhmidwu.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_1, %clamp_max_2, %clamp_max_3, %clamp_max_4, %slice_31, %slice_34, %slice_38, %clamp_max_8], 1), kwargs = {})
triton_poi_fused_cat_27 = async_compile.triton('triton_poi_fused_cat_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 40, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x4 = xindex // 128
    x3 = xindex // 32768
    x5 = ((xindex // 128) % 256)
    x1 = ((xindex // 128) % 16)
    x6 = xindex // 2048
    x2 = ((xindex // 2048) % 16)
    x7 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = 6.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 32, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr5 + (16*x4 + ((-16) + x0)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr6 + ((-16) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 - tmp32
    tmp34 = tl.load(in_ptr7 + ((-16) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp43 = tl.load(in_ptr8 + ((-16) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-16) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 6.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 48, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = tmp53 & tmp55
    tmp57 = tl.load(in_ptr10 + (16*x4 + ((-32) + x0)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr11 + ((-32) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 - tmp58
    tmp60 = tl.load(in_ptr12 + ((-32) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp61 = 1e-05
    tmp62 = tmp60 + tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = tl.full([1], 1, tl.int32)
    tmp65 = tmp64 / tmp63
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tmp68 = tmp59 * tmp67
    tmp69 = tl.load(in_ptr13 + ((-32) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.load(in_ptr14 + ((-32) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 + tmp71
    tmp73 = 0.0
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = 6.0
    tmp76 = triton_helpers.minimum(tmp74, tmp75)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp56, tmp76, tmp77)
    tmp79 = tmp0 >= tmp54
    tmp80 = tl.full([1], 64, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tmp79 & tmp81
    tmp83 = tl.load(in_ptr15 + (16*x4 + ((-48) + x0)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.load(in_ptr16 + ((-48) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 - tmp84
    tmp86 = tl.load(in_ptr17 + ((-48) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp87 = 1e-05
    tmp88 = tmp86 + tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = tl.full([1], 1, tl.int32)
    tmp91 = tmp90 / tmp89
    tmp92 = 1.0
    tmp93 = tmp91 * tmp92
    tmp94 = tmp85 * tmp93
    tmp95 = tl.load(in_ptr18 + ((-48) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp96 = tmp94 * tmp95
    tmp97 = tl.load(in_ptr19 + ((-48) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp96 + tmp97
    tmp99 = 0.0
    tmp100 = triton_helpers.maximum(tmp98, tmp99)
    tmp101 = 6.0
    tmp102 = triton_helpers.minimum(tmp100, tmp101)
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp82, tmp102, tmp103)
    tmp105 = tmp0 >= tmp80
    tmp106 = tl.full([1], 80, tl.int64)
    tmp107 = tmp0 < tmp106
    tmp108 = tmp105 & tmp107
    tmp109 = tl.load(in_ptr20 + (16*x5 + 4352*x3 + ((-64) + x0)), tmp108, eviction_policy='evict_last', other=0.0)
    tmp110 = tl.load(in_ptr21 + ((-64) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp109 - tmp110
    tmp112 = tl.load(in_ptr22 + ((-64) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp113 = 1e-05
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.sqrt(tmp114)
    tmp116 = tl.full([1], 1, tl.int32)
    tmp117 = tmp116 / tmp115
    tmp118 = 1.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp111 * tmp119
    tmp121 = tl.load(in_ptr23 + ((-64) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.load(in_ptr24 + ((-64) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp124 = tmp122 + tmp123
    tmp125 = 0.0
    tmp126 = triton_helpers.maximum(tmp124, tmp125)
    tmp127 = 6.0
    tmp128 = triton_helpers.minimum(tmp126, tmp127)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp108, tmp128, tmp129)
    tmp131 = tmp0 >= tmp106
    tmp132 = tl.full([1], 96, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr25 + (16*x1 + 272*x6 + ((-80) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr26 + ((-80) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp137 = tmp135 - tmp136
    tmp138 = tl.load(in_ptr27 + ((-80) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp139 = 1e-05
    tmp140 = tmp138 + tmp139
    tmp141 = libdevice.sqrt(tmp140)
    tmp142 = tl.full([1], 1, tl.int32)
    tmp143 = tmp142 / tmp141
    tmp144 = 1.0
    tmp145 = tmp143 * tmp144
    tmp146 = tmp137 * tmp145
    tmp147 = tl.load(in_ptr28 + ((-80) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp148 = tmp146 * tmp147
    tmp149 = tl.load(in_ptr29 + ((-80) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp148 + tmp149
    tmp151 = 0.0
    tmp152 = triton_helpers.maximum(tmp150, tmp151)
    tmp153 = 6.0
    tmp154 = triton_helpers.minimum(tmp152, tmp153)
    tmp155 = tl.full(tmp154.shape, 0.0, tmp154.dtype)
    tmp156 = tl.where(tmp134, tmp154, tmp155)
    tmp157 = tmp0 >= tmp132
    tmp158 = tl.full([1], 112, tl.int64)
    tmp159 = tmp0 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tl.load(in_ptr30 + (16*x1 + 272*x2 + 4624*x3 + ((-96) + x0)), tmp160, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr31 + ((-96) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp163 = tmp161 - tmp162
    tmp164 = tl.load(in_ptr32 + ((-96) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp165 = 1e-05
    tmp166 = tmp164 + tmp165
    tmp167 = libdevice.sqrt(tmp166)
    tmp168 = tl.full([1], 1, tl.int32)
    tmp169 = tmp168 / tmp167
    tmp170 = 1.0
    tmp171 = tmp169 * tmp170
    tmp172 = tmp163 * tmp171
    tmp173 = tl.load(in_ptr33 + ((-96) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp174 = tmp172 * tmp173
    tmp175 = tl.load(in_ptr34 + ((-96) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp176 = tmp174 + tmp175
    tmp177 = 0.0
    tmp178 = triton_helpers.maximum(tmp176, tmp177)
    tmp179 = 6.0
    tmp180 = triton_helpers.minimum(tmp178, tmp179)
    tmp181 = tl.full(tmp180.shape, 0.0, tmp180.dtype)
    tmp182 = tl.where(tmp160, tmp180, tmp181)
    tmp183 = tmp0 >= tmp158
    tmp184 = tl.full([1], 128, tl.int64)
    tmp185 = tmp0 < tmp184
    tmp186 = tl.load(in_ptr35 + (16*x4 + ((-112) + x0)), tmp183, eviction_policy='evict_last', other=0.0)
    tmp187 = tl.load(in_ptr36 + ((-112) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp188 = tmp186 - tmp187
    tmp189 = tl.load(in_ptr37 + ((-112) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp190 = 1e-05
    tmp191 = tmp189 + tmp190
    tmp192 = libdevice.sqrt(tmp191)
    tmp193 = tl.full([1], 1, tl.int32)
    tmp194 = tmp193 / tmp192
    tmp195 = 1.0
    tmp196 = tmp194 * tmp195
    tmp197 = tmp188 * tmp196
    tmp198 = tl.load(in_ptr38 + ((-112) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp199 = tmp197 * tmp198
    tmp200 = tl.load(in_ptr39 + ((-112) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp201 = tmp199 + tmp200
    tmp202 = 0.0
    tmp203 = triton_helpers.maximum(tmp201, tmp202)
    tmp204 = 6.0
    tmp205 = triton_helpers.minimum(tmp203, tmp204)
    tmp206 = tl.full(tmp205.shape, 0.0, tmp205.dtype)
    tmp207 = tl.where(tmp183, tmp205, tmp206)
    tmp208 = tl.where(tmp160, tmp182, tmp207)
    tmp209 = tl.where(tmp134, tmp156, tmp208)
    tmp210 = tl.where(tmp108, tmp130, tmp209)
    tmp211 = tl.where(tmp82, tmp104, tmp210)
    tmp212 = tl.where(tmp56, tmp78, tmp211)
    tmp213 = tl.where(tmp30, tmp52, tmp212)
    tmp214 = tl.where(tmp4, tmp26, tmp213)
    tl.store(out_ptr0 + (x7), tmp214, None)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crzjz3dyyjicwy6ovdxyfivpusyipvdggwuwzmg2xkrq6lru5oz2.py
# Topologically Sorted Source Nodes: [input_30, input_31], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_30 => convolution_10
#   input_31 => add_21, mul_31, mul_32, sub_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_62, %primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_21 : [num_users=9] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/tq/ctq46go23zhowlxk3vlkjw4cbbgwzn2kapesq3uv32vm6ng3u4g6.py
# Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_32 => convolution_11
# Graph fragment:
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_21, %primals_68, %primals_69, [1, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_29 = async_compile.triton('triton_poi_fused_convolution_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_29(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/vc/cvc3ghkzherd57ovlnd6nd2cdtilqrkylfg3wwojn7es25hiu437.py
# Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_44 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_21, %primals_92, %primals_93, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_30 = async_compile.triton('triton_poi_fused_convolution_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_30(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6q/c6qlv5a5kqodfamn52feenljq23x2tmqiibs7pz4qqutuo7swrhb.py
# Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_50 => convolution_17
# Graph fragment:
#   %convolution_17 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_21, %primals_104, %primals_105, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rw/crwo4o7feiao625km6oamdpejuzfqhzrlza76ehchxvzlpdnhn4m.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_9, %clamp_max_10, %clamp_max_11, %clamp_max_12, %slice_53, %slice_56, %slice_60, %clamp_max_16], 1), kwargs = {})
triton_poi_fused_cat_32 = async_compile.triton('triton_poi_fused_cat_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 40, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 192)
    x4 = xindex // 192
    x3 = xindex // 49152
    x5 = ((xindex // 192) % 256)
    x1 = ((xindex // 192) % 16)
    x6 = xindex // 3072
    x2 = ((xindex // 3072) % 16)
    x7 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (24*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = 6.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 48, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr5 + (24*x4 + ((-24) + x0)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr6 + ((-24) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 - tmp32
    tmp34 = tl.load(in_ptr7 + ((-24) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp43 = tl.load(in_ptr8 + ((-24) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-24) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 6.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 72, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = tmp53 & tmp55
    tmp57 = tl.load(in_ptr10 + (24*x4 + ((-48) + x0)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr11 + ((-48) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 - tmp58
    tmp60 = tl.load(in_ptr12 + ((-48) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp61 = 1e-05
    tmp62 = tmp60 + tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = tl.full([1], 1, tl.int32)
    tmp65 = tmp64 / tmp63
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tmp68 = tmp59 * tmp67
    tmp69 = tl.load(in_ptr13 + ((-48) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.load(in_ptr14 + ((-48) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 + tmp71
    tmp73 = 0.0
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = 6.0
    tmp76 = triton_helpers.minimum(tmp74, tmp75)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp56, tmp76, tmp77)
    tmp79 = tmp0 >= tmp54
    tmp80 = tl.full([1], 96, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tmp79 & tmp81
    tmp83 = tl.load(in_ptr15 + (24*x4 + ((-72) + x0)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.load(in_ptr16 + ((-72) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 - tmp84
    tmp86 = tl.load(in_ptr17 + ((-72) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp87 = 1e-05
    tmp88 = tmp86 + tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = tl.full([1], 1, tl.int32)
    tmp91 = tmp90 / tmp89
    tmp92 = 1.0
    tmp93 = tmp91 * tmp92
    tmp94 = tmp85 * tmp93
    tmp95 = tl.load(in_ptr18 + ((-72) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp96 = tmp94 * tmp95
    tmp97 = tl.load(in_ptr19 + ((-72) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp96 + tmp97
    tmp99 = 0.0
    tmp100 = triton_helpers.maximum(tmp98, tmp99)
    tmp101 = 6.0
    tmp102 = triton_helpers.minimum(tmp100, tmp101)
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp82, tmp102, tmp103)
    tmp105 = tmp0 >= tmp80
    tmp106 = tl.full([1], 120, tl.int64)
    tmp107 = tmp0 < tmp106
    tmp108 = tmp105 & tmp107
    tmp109 = tl.load(in_ptr20 + (24*x5 + 6528*x3 + ((-96) + x0)), tmp108, eviction_policy='evict_last', other=0.0)
    tmp110 = tl.load(in_ptr21 + ((-96) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp109 - tmp110
    tmp112 = tl.load(in_ptr22 + ((-96) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp113 = 1e-05
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.sqrt(tmp114)
    tmp116 = tl.full([1], 1, tl.int32)
    tmp117 = tmp116 / tmp115
    tmp118 = 1.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp111 * tmp119
    tmp121 = tl.load(in_ptr23 + ((-96) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.load(in_ptr24 + ((-96) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp124 = tmp122 + tmp123
    tmp125 = 0.0
    tmp126 = triton_helpers.maximum(tmp124, tmp125)
    tmp127 = 6.0
    tmp128 = triton_helpers.minimum(tmp126, tmp127)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp108, tmp128, tmp129)
    tmp131 = tmp0 >= tmp106
    tmp132 = tl.full([1], 144, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr25 + (24*x1 + 408*x6 + ((-120) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr26 + ((-120) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp137 = tmp135 - tmp136
    tmp138 = tl.load(in_ptr27 + ((-120) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp139 = 1e-05
    tmp140 = tmp138 + tmp139
    tmp141 = libdevice.sqrt(tmp140)
    tmp142 = tl.full([1], 1, tl.int32)
    tmp143 = tmp142 / tmp141
    tmp144 = 1.0
    tmp145 = tmp143 * tmp144
    tmp146 = tmp137 * tmp145
    tmp147 = tl.load(in_ptr28 + ((-120) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp148 = tmp146 * tmp147
    tmp149 = tl.load(in_ptr29 + ((-120) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp148 + tmp149
    tmp151 = 0.0
    tmp152 = triton_helpers.maximum(tmp150, tmp151)
    tmp153 = 6.0
    tmp154 = triton_helpers.minimum(tmp152, tmp153)
    tmp155 = tl.full(tmp154.shape, 0.0, tmp154.dtype)
    tmp156 = tl.where(tmp134, tmp154, tmp155)
    tmp157 = tmp0 >= tmp132
    tmp158 = tl.full([1], 168, tl.int64)
    tmp159 = tmp0 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tl.load(in_ptr30 + (24*x1 + 408*x2 + 6936*x3 + ((-144) + x0)), tmp160, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr31 + ((-144) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp163 = tmp161 - tmp162
    tmp164 = tl.load(in_ptr32 + ((-144) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp165 = 1e-05
    tmp166 = tmp164 + tmp165
    tmp167 = libdevice.sqrt(tmp166)
    tmp168 = tl.full([1], 1, tl.int32)
    tmp169 = tmp168 / tmp167
    tmp170 = 1.0
    tmp171 = tmp169 * tmp170
    tmp172 = tmp163 * tmp171
    tmp173 = tl.load(in_ptr33 + ((-144) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp174 = tmp172 * tmp173
    tmp175 = tl.load(in_ptr34 + ((-144) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp176 = tmp174 + tmp175
    tmp177 = 0.0
    tmp178 = triton_helpers.maximum(tmp176, tmp177)
    tmp179 = 6.0
    tmp180 = triton_helpers.minimum(tmp178, tmp179)
    tmp181 = tl.full(tmp180.shape, 0.0, tmp180.dtype)
    tmp182 = tl.where(tmp160, tmp180, tmp181)
    tmp183 = tmp0 >= tmp158
    tmp184 = tl.full([1], 192, tl.int64)
    tmp185 = tmp0 < tmp184
    tmp186 = tl.load(in_ptr35 + (24*x4 + ((-168) + x0)), tmp183, eviction_policy='evict_last', other=0.0)
    tmp187 = tl.load(in_ptr36 + ((-168) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp188 = tmp186 - tmp187
    tmp189 = tl.load(in_ptr37 + ((-168) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp190 = 1e-05
    tmp191 = tmp189 + tmp190
    tmp192 = libdevice.sqrt(tmp191)
    tmp193 = tl.full([1], 1, tl.int32)
    tmp194 = tmp193 / tmp192
    tmp195 = 1.0
    tmp196 = tmp194 * tmp195
    tmp197 = tmp188 * tmp196
    tmp198 = tl.load(in_ptr38 + ((-168) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp199 = tmp197 * tmp198
    tmp200 = tl.load(in_ptr39 + ((-168) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp201 = tmp199 + tmp200
    tmp202 = 0.0
    tmp203 = triton_helpers.maximum(tmp201, tmp202)
    tmp204 = 6.0
    tmp205 = triton_helpers.minimum(tmp203, tmp204)
    tmp206 = tl.full(tmp205.shape, 0.0, tmp205.dtype)
    tmp207 = tl.where(tmp183, tmp205, tmp206)
    tmp208 = tl.where(tmp160, tmp182, tmp207)
    tmp209 = tl.where(tmp134, tmp156, tmp208)
    tmp210 = tl.where(tmp108, tmp130, tmp209)
    tmp211 = tl.where(tmp82, tmp104, tmp210)
    tmp212 = tl.where(tmp56, tmp78, tmp211)
    tmp213 = tl.where(tmp30, tmp52, tmp212)
    tmp214 = tl.where(tmp4, tmp26, tmp213)
    tl.store(out_ptr0 + (x7), tmp214, None)
''', device_str='cuda')


# kernel path: inductor_cache/qz/cqz4o23iovof5uubnqdry24x3rokp3frih6gcy7nhv5pn6nfhcxn.py
# Topologically Sorted Source Nodes: [input_57, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   input_57 => add_39, mul_58, mul_59, sub_19
#   x => constant_pad_nd
# Graph fragment:
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_153), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_157), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_159), kwargs = {})
#   %constant_pad_nd : [num_users=9] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_39, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 408) % 17)
    x1 = ((xindex // 24) % 17)
    x3 = xindex // 6936
    x4 = (xindex % 408)
    x0 = (xindex % 24)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + 384*x2 + 6144*x3), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 - tmp7
    tmp9 = tl.load(in_ptr2 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp18 = tl.load(in_ptr3 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr4 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tl.store(out_ptr0 + (x5), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zc/czcv3tcjuqypb7otek27w7zmh4s4wwbhqp4r73sksb45vyg4kdbs.py
# Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_58 => convolution_20
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %primals_122, %primals_123, [2, 2], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_34 = async_compile.triton('triton_poi_fused_convolution_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_34(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xy/cxyxo7at7frupxco27wzrjh2yjxpudeyi7wk664zgh32h3ab5hhs.py
# Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_96, %slice_100, %slice_104, %slice_108, %slice_112, %slice_116, %slice_120, %slice_124], 1), kwargs = {})
triton_poi_fused_cat_35 = async_compile.triton('triton_poi_fused_cat_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 40, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 8)
    x2 = ((xindex // 1536) % 8)
    x3 = xindex // 12288
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (24*x1 + 216*x2 + 1944*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = 6.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 48, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr5 + (24*x1 + 216*x2 + 1944*x3 + ((-24) + x0)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr6 + ((-24) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 - tmp32
    tmp34 = tl.load(in_ptr7 + ((-24) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp43 = tl.load(in_ptr8 + ((-24) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-24) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 6.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 72, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = tmp53 & tmp55
    tmp57 = tl.load(in_ptr10 + (24*x1 + 216*x2 + 1944*x3 + ((-48) + x0)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr11 + ((-48) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 - tmp58
    tmp60 = tl.load(in_ptr12 + ((-48) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp61 = 1e-05
    tmp62 = tmp60 + tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = tl.full([1], 1, tl.int32)
    tmp65 = tmp64 / tmp63
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tmp68 = tmp59 * tmp67
    tmp69 = tl.load(in_ptr13 + ((-48) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.load(in_ptr14 + ((-48) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 + tmp71
    tmp73 = 0.0
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = 6.0
    tmp76 = triton_helpers.minimum(tmp74, tmp75)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp56, tmp76, tmp77)
    tmp79 = tmp0 >= tmp54
    tmp80 = tl.full([1], 96, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tmp79 & tmp81
    tmp83 = tl.load(in_ptr15 + (24*x1 + 216*x2 + 1944*x3 + ((-72) + x0)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.load(in_ptr16 + ((-72) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 - tmp84
    tmp86 = tl.load(in_ptr17 + ((-72) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp87 = 1e-05
    tmp88 = tmp86 + tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = tl.full([1], 1, tl.int32)
    tmp91 = tmp90 / tmp89
    tmp92 = 1.0
    tmp93 = tmp91 * tmp92
    tmp94 = tmp85 * tmp93
    tmp95 = tl.load(in_ptr18 + ((-72) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp96 = tmp94 * tmp95
    tmp97 = tl.load(in_ptr19 + ((-72) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp96 + tmp97
    tmp99 = 0.0
    tmp100 = triton_helpers.maximum(tmp98, tmp99)
    tmp101 = 6.0
    tmp102 = triton_helpers.minimum(tmp100, tmp101)
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp82, tmp102, tmp103)
    tmp105 = tmp0 >= tmp80
    tmp106 = tl.full([1], 120, tl.int64)
    tmp107 = tmp0 < tmp106
    tmp108 = tmp105 & tmp107
    tmp109 = tl.load(in_ptr20 + (24*x1 + 216*x2 + 1944*x3 + ((-96) + x0)), tmp108, eviction_policy='evict_last', other=0.0)
    tmp110 = tl.load(in_ptr21 + ((-96) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp109 - tmp110
    tmp112 = tl.load(in_ptr22 + ((-96) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp113 = 1e-05
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.sqrt(tmp114)
    tmp116 = tl.full([1], 1, tl.int32)
    tmp117 = tmp116 / tmp115
    tmp118 = 1.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp111 * tmp119
    tmp121 = tl.load(in_ptr23 + ((-96) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.load(in_ptr24 + ((-96) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp124 = tmp122 + tmp123
    tmp125 = 0.0
    tmp126 = triton_helpers.maximum(tmp124, tmp125)
    tmp127 = 6.0
    tmp128 = triton_helpers.minimum(tmp126, tmp127)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp108, tmp128, tmp129)
    tmp131 = tmp0 >= tmp106
    tmp132 = tl.full([1], 144, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr25 + (24*x1 + 216*x2 + 1944*x3 + ((-120) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr26 + ((-120) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp137 = tmp135 - tmp136
    tmp138 = tl.load(in_ptr27 + ((-120) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp139 = 1e-05
    tmp140 = tmp138 + tmp139
    tmp141 = libdevice.sqrt(tmp140)
    tmp142 = tl.full([1], 1, tl.int32)
    tmp143 = tmp142 / tmp141
    tmp144 = 1.0
    tmp145 = tmp143 * tmp144
    tmp146 = tmp137 * tmp145
    tmp147 = tl.load(in_ptr28 + ((-120) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp148 = tmp146 * tmp147
    tmp149 = tl.load(in_ptr29 + ((-120) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp148 + tmp149
    tmp151 = 0.0
    tmp152 = triton_helpers.maximum(tmp150, tmp151)
    tmp153 = 6.0
    tmp154 = triton_helpers.minimum(tmp152, tmp153)
    tmp155 = tl.full(tmp154.shape, 0.0, tmp154.dtype)
    tmp156 = tl.where(tmp134, tmp154, tmp155)
    tmp157 = tmp0 >= tmp132
    tmp158 = tl.full([1], 168, tl.int64)
    tmp159 = tmp0 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tl.load(in_ptr30 + (24*x1 + 216*x2 + 1944*x3 + ((-144) + x0)), tmp160, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr31 + ((-144) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp163 = tmp161 - tmp162
    tmp164 = tl.load(in_ptr32 + ((-144) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp165 = 1e-05
    tmp166 = tmp164 + tmp165
    tmp167 = libdevice.sqrt(tmp166)
    tmp168 = tl.full([1], 1, tl.int32)
    tmp169 = tmp168 / tmp167
    tmp170 = 1.0
    tmp171 = tmp169 * tmp170
    tmp172 = tmp163 * tmp171
    tmp173 = tl.load(in_ptr33 + ((-144) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp174 = tmp172 * tmp173
    tmp175 = tl.load(in_ptr34 + ((-144) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp176 = tmp174 + tmp175
    tmp177 = 0.0
    tmp178 = triton_helpers.maximum(tmp176, tmp177)
    tmp179 = 6.0
    tmp180 = triton_helpers.minimum(tmp178, tmp179)
    tmp181 = tl.full(tmp180.shape, 0.0, tmp180.dtype)
    tmp182 = tl.where(tmp160, tmp180, tmp181)
    tmp183 = tmp0 >= tmp158
    tmp184 = tl.full([1], 192, tl.int64)
    tmp185 = tmp0 < tmp184
    tmp186 = tl.load(in_ptr35 + (24*x1 + 216*x2 + 1944*x3 + ((-168) + x0)), tmp183, eviction_policy='evict_last', other=0.0)
    tmp187 = tl.load(in_ptr36 + ((-168) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp188 = tmp186 - tmp187
    tmp189 = tl.load(in_ptr37 + ((-168) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp190 = 1e-05
    tmp191 = tmp189 + tmp190
    tmp192 = libdevice.sqrt(tmp191)
    tmp193 = tl.full([1], 1, tl.int32)
    tmp194 = tmp193 / tmp192
    tmp195 = 1.0
    tmp196 = tmp194 * tmp195
    tmp197 = tmp188 * tmp196
    tmp198 = tl.load(in_ptr38 + ((-168) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp199 = tmp197 * tmp198
    tmp200 = tl.load(in_ptr39 + ((-168) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp201 = tmp199 + tmp200
    tmp202 = 0.0
    tmp203 = triton_helpers.maximum(tmp201, tmp202)
    tmp204 = 6.0
    tmp205 = triton_helpers.minimum(tmp203, tmp204)
    tmp206 = tl.full(tmp205.shape, 0.0, tmp205.dtype)
    tmp207 = tl.where(tmp183, tmp205, tmp206)
    tmp208 = tl.where(tmp160, tmp182, tmp207)
    tmp209 = tl.where(tmp134, tmp156, tmp208)
    tmp210 = tl.where(tmp108, tmp130, tmp209)
    tmp211 = tl.where(tmp82, tmp104, tmp210)
    tmp212 = tl.where(tmp56, tmp78, tmp211)
    tmp213 = tl.where(tmp30, tmp52, tmp212)
    tmp214 = tl.where(tmp4, tmp26, tmp213)
    tl.store(out_ptr0 + (x5), tmp214, None)
''', device_str='cuda')


# kernel path: inductor_cache/bq/cbqhrxb6vhzbff3pkwo3dmruro4etugt227lsf7ukgieaag5bb3l.py
# Topologically Sorted Source Nodes: [input_82, input_83], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_82 => convolution_28
#   input_83 => add_57, mul_85, mul_86, sub_28
# Graph fragment:
#   %convolution_28 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %primals_170, %primals_171, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %add_57 : [num_users=9] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/kt/ckthv2tiiqvoqosaalk6btlwbuh64enpsgcdiakyk5ulkx34sifb.py
# Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_84 => convolution_29
# Graph fragment:
#   %convolution_29 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_57, %primals_176, %primals_177, [1, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_37 = async_compile.triton('triton_poi_fused_convolution_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_37(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/s6/cs6ixo3dtmdysmhizibd76hsvotfzalnrm7t4f2ax6mhypibyks3.py
# Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_96 => convolution_33
# Graph fragment:
#   %convolution_33 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_57, %primals_200, %primals_201, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ed/cedd3roepj56mdfqdvefmzkfqofpdkiksl7wqv6wymf5o6gfppyd.py
# Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_102 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_57, %primals_212, %primals_213, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_39 = async_compile.triton('triton_poi_fused_convolution_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_39(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u4/cu4y4s4wvt3xir362q7ljkjfckerwya5gglv32moyscyjhys7tpl.py
# Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_25, %clamp_max_26, %clamp_max_27, %clamp_max_28, %slice_137, %slice_140, %slice_144, %clamp_max_32], 1), kwargs = {})
triton_poi_fused_cat_40 = async_compile.triton('triton_poi_fused_cat_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 40, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x4 = xindex // 256
    x3 = xindex // 16384
    x5 = ((xindex // 256) % 64)
    x1 = ((xindex // 256) % 8)
    x6 = xindex // 2048
    x2 = ((xindex // 2048) % 8)
    x7 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = 6.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 64, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr5 + (32*x4 + ((-32) + x0)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr6 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 - tmp32
    tmp34 = tl.load(in_ptr7 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp43 = tl.load(in_ptr8 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 6.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 96, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = tmp53 & tmp55
    tmp57 = tl.load(in_ptr10 + (32*x4 + ((-64) + x0)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr11 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 - tmp58
    tmp60 = tl.load(in_ptr12 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp61 = 1e-05
    tmp62 = tmp60 + tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = tl.full([1], 1, tl.int32)
    tmp65 = tmp64 / tmp63
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tmp68 = tmp59 * tmp67
    tmp69 = tl.load(in_ptr13 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.load(in_ptr14 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 + tmp71
    tmp73 = 0.0
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = 6.0
    tmp76 = triton_helpers.minimum(tmp74, tmp75)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp56, tmp76, tmp77)
    tmp79 = tmp0 >= tmp54
    tmp80 = tl.full([1], 128, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tmp79 & tmp81
    tmp83 = tl.load(in_ptr15 + (32*x4 + ((-96) + x0)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.load(in_ptr16 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 - tmp84
    tmp86 = tl.load(in_ptr17 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp87 = 1e-05
    tmp88 = tmp86 + tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = tl.full([1], 1, tl.int32)
    tmp91 = tmp90 / tmp89
    tmp92 = 1.0
    tmp93 = tmp91 * tmp92
    tmp94 = tmp85 * tmp93
    tmp95 = tl.load(in_ptr18 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp96 = tmp94 * tmp95
    tmp97 = tl.load(in_ptr19 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp96 + tmp97
    tmp99 = 0.0
    tmp100 = triton_helpers.maximum(tmp98, tmp99)
    tmp101 = 6.0
    tmp102 = triton_helpers.minimum(tmp100, tmp101)
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp82, tmp102, tmp103)
    tmp105 = tmp0 >= tmp80
    tmp106 = tl.full([1], 160, tl.int64)
    tmp107 = tmp0 < tmp106
    tmp108 = tmp105 & tmp107
    tmp109 = tl.load(in_ptr20 + (32*x5 + 2304*x3 + ((-128) + x0)), tmp108, eviction_policy='evict_last', other=0.0)
    tmp110 = tl.load(in_ptr21 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp109 - tmp110
    tmp112 = tl.load(in_ptr22 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp113 = 1e-05
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.sqrt(tmp114)
    tmp116 = tl.full([1], 1, tl.int32)
    tmp117 = tmp116 / tmp115
    tmp118 = 1.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp111 * tmp119
    tmp121 = tl.load(in_ptr23 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.load(in_ptr24 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp124 = tmp122 + tmp123
    tmp125 = 0.0
    tmp126 = triton_helpers.maximum(tmp124, tmp125)
    tmp127 = 6.0
    tmp128 = triton_helpers.minimum(tmp126, tmp127)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp108, tmp128, tmp129)
    tmp131 = tmp0 >= tmp106
    tmp132 = tl.full([1], 192, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr25 + (32*x1 + 288*x6 + ((-160) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr26 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp137 = tmp135 - tmp136
    tmp138 = tl.load(in_ptr27 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp139 = 1e-05
    tmp140 = tmp138 + tmp139
    tmp141 = libdevice.sqrt(tmp140)
    tmp142 = tl.full([1], 1, tl.int32)
    tmp143 = tmp142 / tmp141
    tmp144 = 1.0
    tmp145 = tmp143 * tmp144
    tmp146 = tmp137 * tmp145
    tmp147 = tl.load(in_ptr28 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp148 = tmp146 * tmp147
    tmp149 = tl.load(in_ptr29 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp148 + tmp149
    tmp151 = 0.0
    tmp152 = triton_helpers.maximum(tmp150, tmp151)
    tmp153 = 6.0
    tmp154 = triton_helpers.minimum(tmp152, tmp153)
    tmp155 = tl.full(tmp154.shape, 0.0, tmp154.dtype)
    tmp156 = tl.where(tmp134, tmp154, tmp155)
    tmp157 = tmp0 >= tmp132
    tmp158 = tl.full([1], 224, tl.int64)
    tmp159 = tmp0 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tl.load(in_ptr30 + (32*x1 + 288*x2 + 2592*x3 + ((-192) + x0)), tmp160, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr31 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp163 = tmp161 - tmp162
    tmp164 = tl.load(in_ptr32 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp165 = 1e-05
    tmp166 = tmp164 + tmp165
    tmp167 = libdevice.sqrt(tmp166)
    tmp168 = tl.full([1], 1, tl.int32)
    tmp169 = tmp168 / tmp167
    tmp170 = 1.0
    tmp171 = tmp169 * tmp170
    tmp172 = tmp163 * tmp171
    tmp173 = tl.load(in_ptr33 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp174 = tmp172 * tmp173
    tmp175 = tl.load(in_ptr34 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp176 = tmp174 + tmp175
    tmp177 = 0.0
    tmp178 = triton_helpers.maximum(tmp176, tmp177)
    tmp179 = 6.0
    tmp180 = triton_helpers.minimum(tmp178, tmp179)
    tmp181 = tl.full(tmp180.shape, 0.0, tmp180.dtype)
    tmp182 = tl.where(tmp160, tmp180, tmp181)
    tmp183 = tmp0 >= tmp158
    tmp184 = tl.full([1], 256, tl.int64)
    tmp185 = tmp0 < tmp184
    tmp186 = tl.load(in_ptr35 + (32*x4 + ((-224) + x0)), tmp183, eviction_policy='evict_last', other=0.0)
    tmp187 = tl.load(in_ptr36 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp188 = tmp186 - tmp187
    tmp189 = tl.load(in_ptr37 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp190 = 1e-05
    tmp191 = tmp189 + tmp190
    tmp192 = libdevice.sqrt(tmp191)
    tmp193 = tl.full([1], 1, tl.int32)
    tmp194 = tmp193 / tmp192
    tmp195 = 1.0
    tmp196 = tmp194 * tmp195
    tmp197 = tmp188 * tmp196
    tmp198 = tl.load(in_ptr38 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp199 = tmp197 * tmp198
    tmp200 = tl.load(in_ptr39 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp201 = tmp199 + tmp200
    tmp202 = 0.0
    tmp203 = triton_helpers.maximum(tmp201, tmp202)
    tmp204 = 6.0
    tmp205 = triton_helpers.minimum(tmp203, tmp204)
    tmp206 = tl.full(tmp205.shape, 0.0, tmp205.dtype)
    tmp207 = tl.where(tmp183, tmp205, tmp206)
    tmp208 = tl.where(tmp160, tmp182, tmp207)
    tmp209 = tl.where(tmp134, tmp156, tmp208)
    tmp210 = tl.where(tmp108, tmp130, tmp209)
    tmp211 = tl.where(tmp82, tmp104, tmp210)
    tmp212 = tl.where(tmp56, tmp78, tmp211)
    tmp213 = tl.where(tmp30, tmp52, tmp212)
    tmp214 = tl.where(tmp4, tmp26, tmp213)
    tl.store(out_ptr0 + (x7), tmp214, None)
''', device_str='cuda')


# kernel path: inductor_cache/vk/cvkzgni6zqqakimhxdfsudt6xklgkm6r27vcpnyq5cud54n27wdn.py
# Topologically Sorted Source Nodes: [input_109, input_110, input_113, input_116, input_119, input_122, input_125, input_128, input_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
# Source node to ATen node mapping:
#   input_109 => add_75, mul_112, mul_113, sub_37
#   input_110 => convolution_38
#   input_113 => convolution_39
#   input_116 => convolution_40
#   input_119 => convolution_41
#   input_122 => convolution_42
#   input_125 => convolution_43
#   input_128 => convolution_44
#   input_131 => convolution_45
# Graph fragment:
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_297), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %add_75 : [num_users=9] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %convolution_38 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_230, %primals_231, [2, 2], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_39 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_236, %primals_237, [2, 2], [0, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_40 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_242, %primals_243, [2, 2], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_41 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_248, %primals_249, [2, 2], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_42 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_254, %primals_255, [2, 2], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_43 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_260, %primals_261, [2, 2], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_44 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_266, %primals_267, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_45 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_272, %primals_273, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32*x2 + 2048*y1), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 64*y3), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 32*x2 + 2048*y1), tmp15, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 32*x2 + 2048*y1), tmp15, xmask & ymask)
    tl.store(out_ptr3 + (y0 + 32*x2 + 2048*y1), tmp15, xmask & ymask)
    tl.store(out_ptr4 + (y0 + 32*x2 + 2048*y1), tmp15, xmask & ymask)
    tl.store(out_ptr5 + (y0 + 32*x2 + 2048*y1), tmp15, xmask & ymask)
    tl.store(out_ptr6 + (y0 + 32*x2 + 2048*y1), tmp15, xmask & ymask)
    tl.store(out_ptr7 + (y0 + 32*x2 + 2048*y1), tmp15, xmask & ymask)
    tl.store(out_ptr8 + (y0 + 32*x2 + 2048*y1), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ew/cewktbkeb2fpyfylh5prvgxlcjfppmy2sf7zscxxoiceuexs3pgr.py
# Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_110 => convolution_38
# Graph fragment:
#   %convolution_38 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_230, %primals_231, [2, 2], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_42 = async_compile.triton('triton_poi_fused_convolution_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_42(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/en/cenz7suoywfsv3hwuxx2ouxsvf4fofqgwitblmsl7a6potbpuh4e.py
# Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_122 => convolution_42
# Graph fragment:
#   %convolution_42 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_254, %primals_255, [2, 2], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_43 = async_compile.triton('triton_poi_fused_convolution_43', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_43(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r7/cr7tct5aefe5wndwsh5xxkpkmk4eervzjrzyt7w2dzdpty4uvoig.py
# Topologically Sorted Source Nodes: [input_128], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_128 => convolution_44
# Graph fragment:
#   %convolution_44 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_266, %primals_267, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_44 = async_compile.triton('triton_poi_fused_convolution_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_44(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2k/c2kitavjjddo4cfhoj4lsw2uhc2p4h7tco6pgtn7epbgyeumtliv.py
# Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_4 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_33, %clamp_max_34, %clamp_max_35, %clamp_max_36, %slice_175, %slice_178, %slice_182, %clamp_max_40], 1), kwargs = {})
triton_poi_fused_cat_45 = async_compile.triton('triton_poi_fused_cat_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 40, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x4 = xindex // 256
    x3 = xindex // 4096
    x5 = ((xindex // 256) % 16)
    x1 = ((xindex // 256) % 4)
    x6 = xindex // 1024
    x2 = ((xindex // 1024) % 4)
    x7 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = 6.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 64, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr5 + (32*x4 + ((-32) + x0)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr6 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 - tmp32
    tmp34 = tl.load(in_ptr7 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp43 = tl.load(in_ptr8 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 6.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 96, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = tmp53 & tmp55
    tmp57 = tl.load(in_ptr10 + (32*x4 + ((-64) + x0)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr11 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 - tmp58
    tmp60 = tl.load(in_ptr12 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp61 = 1e-05
    tmp62 = tmp60 + tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = tl.full([1], 1, tl.int32)
    tmp65 = tmp64 / tmp63
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tmp68 = tmp59 * tmp67
    tmp69 = tl.load(in_ptr13 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.load(in_ptr14 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 + tmp71
    tmp73 = 0.0
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = 6.0
    tmp76 = triton_helpers.minimum(tmp74, tmp75)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp56, tmp76, tmp77)
    tmp79 = tmp0 >= tmp54
    tmp80 = tl.full([1], 128, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tmp79 & tmp81
    tmp83 = tl.load(in_ptr15 + (32*x4 + ((-96) + x0)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.load(in_ptr16 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 - tmp84
    tmp86 = tl.load(in_ptr17 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp87 = 1e-05
    tmp88 = tmp86 + tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = tl.full([1], 1, tl.int32)
    tmp91 = tmp90 / tmp89
    tmp92 = 1.0
    tmp93 = tmp91 * tmp92
    tmp94 = tmp85 * tmp93
    tmp95 = tl.load(in_ptr18 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp96 = tmp94 * tmp95
    tmp97 = tl.load(in_ptr19 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp96 + tmp97
    tmp99 = 0.0
    tmp100 = triton_helpers.maximum(tmp98, tmp99)
    tmp101 = 6.0
    tmp102 = triton_helpers.minimum(tmp100, tmp101)
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp82, tmp102, tmp103)
    tmp105 = tmp0 >= tmp80
    tmp106 = tl.full([1], 160, tl.int64)
    tmp107 = tmp0 < tmp106
    tmp108 = tmp105 & tmp107
    tmp109 = tl.load(in_ptr20 + (32*x5 + 640*x3 + ((-128) + x0)), tmp108, eviction_policy='evict_last', other=0.0)
    tmp110 = tl.load(in_ptr21 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp109 - tmp110
    tmp112 = tl.load(in_ptr22 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp113 = 1e-05
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.sqrt(tmp114)
    tmp116 = tl.full([1], 1, tl.int32)
    tmp117 = tmp116 / tmp115
    tmp118 = 1.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp111 * tmp119
    tmp121 = tl.load(in_ptr23 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.load(in_ptr24 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp124 = tmp122 + tmp123
    tmp125 = 0.0
    tmp126 = triton_helpers.maximum(tmp124, tmp125)
    tmp127 = 6.0
    tmp128 = triton_helpers.minimum(tmp126, tmp127)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp108, tmp128, tmp129)
    tmp131 = tmp0 >= tmp106
    tmp132 = tl.full([1], 192, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr25 + (32*x1 + 160*x6 + ((-160) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr26 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp137 = tmp135 - tmp136
    tmp138 = tl.load(in_ptr27 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp139 = 1e-05
    tmp140 = tmp138 + tmp139
    tmp141 = libdevice.sqrt(tmp140)
    tmp142 = tl.full([1], 1, tl.int32)
    tmp143 = tmp142 / tmp141
    tmp144 = 1.0
    tmp145 = tmp143 * tmp144
    tmp146 = tmp137 * tmp145
    tmp147 = tl.load(in_ptr28 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp148 = tmp146 * tmp147
    tmp149 = tl.load(in_ptr29 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp148 + tmp149
    tmp151 = 0.0
    tmp152 = triton_helpers.maximum(tmp150, tmp151)
    tmp153 = 6.0
    tmp154 = triton_helpers.minimum(tmp152, tmp153)
    tmp155 = tl.full(tmp154.shape, 0.0, tmp154.dtype)
    tmp156 = tl.where(tmp134, tmp154, tmp155)
    tmp157 = tmp0 >= tmp132
    tmp158 = tl.full([1], 224, tl.int64)
    tmp159 = tmp0 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tl.load(in_ptr30 + (32*x1 + 160*x2 + 800*x3 + ((-192) + x0)), tmp160, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr31 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp163 = tmp161 - tmp162
    tmp164 = tl.load(in_ptr32 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp165 = 1e-05
    tmp166 = tmp164 + tmp165
    tmp167 = libdevice.sqrt(tmp166)
    tmp168 = tl.full([1], 1, tl.int32)
    tmp169 = tmp168 / tmp167
    tmp170 = 1.0
    tmp171 = tmp169 * tmp170
    tmp172 = tmp163 * tmp171
    tmp173 = tl.load(in_ptr33 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp174 = tmp172 * tmp173
    tmp175 = tl.load(in_ptr34 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp176 = tmp174 + tmp175
    tmp177 = 0.0
    tmp178 = triton_helpers.maximum(tmp176, tmp177)
    tmp179 = 6.0
    tmp180 = triton_helpers.minimum(tmp178, tmp179)
    tmp181 = tl.full(tmp180.shape, 0.0, tmp180.dtype)
    tmp182 = tl.where(tmp160, tmp180, tmp181)
    tmp183 = tmp0 >= tmp158
    tmp184 = tl.full([1], 256, tl.int64)
    tmp185 = tmp0 < tmp184
    tmp186 = tl.load(in_ptr35 + (32*x4 + ((-224) + x0)), tmp183, eviction_policy='evict_last', other=0.0)
    tmp187 = tl.load(in_ptr36 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp188 = tmp186 - tmp187
    tmp189 = tl.load(in_ptr37 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp190 = 1e-05
    tmp191 = tmp189 + tmp190
    tmp192 = libdevice.sqrt(tmp191)
    tmp193 = tl.full([1], 1, tl.int32)
    tmp194 = tmp193 / tmp192
    tmp195 = 1.0
    tmp196 = tmp194 * tmp195
    tmp197 = tmp188 * tmp196
    tmp198 = tl.load(in_ptr38 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp199 = tmp197 * tmp198
    tmp200 = tl.load(in_ptr39 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp201 = tmp199 + tmp200
    tmp202 = 0.0
    tmp203 = triton_helpers.maximum(tmp201, tmp202)
    tmp204 = 6.0
    tmp205 = triton_helpers.minimum(tmp203, tmp204)
    tmp206 = tl.full(tmp205.shape, 0.0, tmp205.dtype)
    tmp207 = tl.where(tmp183, tmp205, tmp206)
    tmp208 = tl.where(tmp160, tmp182, tmp207)
    tmp209 = tl.where(tmp134, tmp156, tmp208)
    tmp210 = tl.where(tmp108, tmp130, tmp209)
    tmp211 = tl.where(tmp82, tmp104, tmp210)
    tmp212 = tl.where(tmp56, tmp78, tmp211)
    tmp213 = tl.where(tmp30, tmp52, tmp212)
    tmp214 = tl.where(tmp4, tmp26, tmp213)
    tl.store(out_ptr0 + (x7), tmp214, None)
''', device_str='cuda')


# kernel path: inductor_cache/gg/cggg3sa2xh3rzb4ajpmo7quoyfqwnb4kfg6nx6pewfo2zaaxckmf.py
# Topologically Sorted Source Nodes: [input_135, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   input_135 => add_93, mul_139, mul_140, sub_46
#   x_1 => constant_pad_nd_1
# Graph fragment:
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_369), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %unsqueeze_373), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_140, %unsqueeze_375), kwargs = {})
#   %constant_pad_nd_1 : [num_users=9] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_93, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 160) % 5)
    x1 = ((xindex // 32) % 5)
    x3 = xindex // 800
    x4 = (xindex % 160)
    x0 = (xindex % 32)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + 128*x2 + 512*x3), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 - tmp7
    tmp9 = tl.load(in_ptr2 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp18 = tl.load(in_ptr3 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr4 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tl.store(out_ptr0 + (x5), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/73/c73oxskxazz3hgr6ky37nutzloogttldd2wmzqej6olltxjjxtb5.py
# Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_136 => convolution_47
# Graph fragment:
#   %convolution_47 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %primals_284, %primals_285, [2, 2], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_47 = async_compile.triton('triton_poi_fused_convolution_47', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_47(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/77/c77qmvtup2yv2c5q7rjog4l4u2o6j3jf4aixe7fwginmclysjkq4.py
# Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_5 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_220, %slice_224, %slice_228, %slice_232, %slice_236, %slice_240, %slice_244, %slice_248], 1), kwargs = {})
triton_poi_fused_cat_48 = async_compile.triton('triton_poi_fused_cat_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 40, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 2)
    x2 = ((xindex // 512) % 2)
    x3 = xindex // 1024
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x1 + 96*x2 + 288*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = 6.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 64, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr5 + (32*x1 + 96*x2 + 288*x3 + ((-32) + x0)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr6 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 - tmp32
    tmp34 = tl.load(in_ptr7 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp43 = tl.load(in_ptr8 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 6.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 96, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = tmp53 & tmp55
    tmp57 = tl.load(in_ptr10 + (32*x1 + 96*x2 + 288*x3 + ((-64) + x0)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr11 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 - tmp58
    tmp60 = tl.load(in_ptr12 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp61 = 1e-05
    tmp62 = tmp60 + tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = tl.full([1], 1, tl.int32)
    tmp65 = tmp64 / tmp63
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tmp68 = tmp59 * tmp67
    tmp69 = tl.load(in_ptr13 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.load(in_ptr14 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 + tmp71
    tmp73 = 0.0
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = 6.0
    tmp76 = triton_helpers.minimum(tmp74, tmp75)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp56, tmp76, tmp77)
    tmp79 = tmp0 >= tmp54
    tmp80 = tl.full([1], 128, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tmp79 & tmp81
    tmp83 = tl.load(in_ptr15 + (32*x1 + 96*x2 + 288*x3 + ((-96) + x0)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.load(in_ptr16 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 - tmp84
    tmp86 = tl.load(in_ptr17 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp87 = 1e-05
    tmp88 = tmp86 + tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = tl.full([1], 1, tl.int32)
    tmp91 = tmp90 / tmp89
    tmp92 = 1.0
    tmp93 = tmp91 * tmp92
    tmp94 = tmp85 * tmp93
    tmp95 = tl.load(in_ptr18 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp96 = tmp94 * tmp95
    tmp97 = tl.load(in_ptr19 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp96 + tmp97
    tmp99 = 0.0
    tmp100 = triton_helpers.maximum(tmp98, tmp99)
    tmp101 = 6.0
    tmp102 = triton_helpers.minimum(tmp100, tmp101)
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp82, tmp102, tmp103)
    tmp105 = tmp0 >= tmp80
    tmp106 = tl.full([1], 160, tl.int64)
    tmp107 = tmp0 < tmp106
    tmp108 = tmp105 & tmp107
    tmp109 = tl.load(in_ptr20 + (32*x1 + 96*x2 + 288*x3 + ((-128) + x0)), tmp108, eviction_policy='evict_last', other=0.0)
    tmp110 = tl.load(in_ptr21 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp109 - tmp110
    tmp112 = tl.load(in_ptr22 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp113 = 1e-05
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.sqrt(tmp114)
    tmp116 = tl.full([1], 1, tl.int32)
    tmp117 = tmp116 / tmp115
    tmp118 = 1.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp111 * tmp119
    tmp121 = tl.load(in_ptr23 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.load(in_ptr24 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp124 = tmp122 + tmp123
    tmp125 = 0.0
    tmp126 = triton_helpers.maximum(tmp124, tmp125)
    tmp127 = 6.0
    tmp128 = triton_helpers.minimum(tmp126, tmp127)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp108, tmp128, tmp129)
    tmp131 = tmp0 >= tmp106
    tmp132 = tl.full([1], 192, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr25 + (32*x1 + 96*x2 + 288*x3 + ((-160) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr26 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp137 = tmp135 - tmp136
    tmp138 = tl.load(in_ptr27 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp139 = 1e-05
    tmp140 = tmp138 + tmp139
    tmp141 = libdevice.sqrt(tmp140)
    tmp142 = tl.full([1], 1, tl.int32)
    tmp143 = tmp142 / tmp141
    tmp144 = 1.0
    tmp145 = tmp143 * tmp144
    tmp146 = tmp137 * tmp145
    tmp147 = tl.load(in_ptr28 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp148 = tmp146 * tmp147
    tmp149 = tl.load(in_ptr29 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp148 + tmp149
    tmp151 = 0.0
    tmp152 = triton_helpers.maximum(tmp150, tmp151)
    tmp153 = 6.0
    tmp154 = triton_helpers.minimum(tmp152, tmp153)
    tmp155 = tl.full(tmp154.shape, 0.0, tmp154.dtype)
    tmp156 = tl.where(tmp134, tmp154, tmp155)
    tmp157 = tmp0 >= tmp132
    tmp158 = tl.full([1], 224, tl.int64)
    tmp159 = tmp0 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tl.load(in_ptr30 + (32*x1 + 96*x2 + 288*x3 + ((-192) + x0)), tmp160, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr31 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp163 = tmp161 - tmp162
    tmp164 = tl.load(in_ptr32 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp165 = 1e-05
    tmp166 = tmp164 + tmp165
    tmp167 = libdevice.sqrt(tmp166)
    tmp168 = tl.full([1], 1, tl.int32)
    tmp169 = tmp168 / tmp167
    tmp170 = 1.0
    tmp171 = tmp169 * tmp170
    tmp172 = tmp163 * tmp171
    tmp173 = tl.load(in_ptr33 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp174 = tmp172 * tmp173
    tmp175 = tl.load(in_ptr34 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp176 = tmp174 + tmp175
    tmp177 = 0.0
    tmp178 = triton_helpers.maximum(tmp176, tmp177)
    tmp179 = 6.0
    tmp180 = triton_helpers.minimum(tmp178, tmp179)
    tmp181 = tl.full(tmp180.shape, 0.0, tmp180.dtype)
    tmp182 = tl.where(tmp160, tmp180, tmp181)
    tmp183 = tmp0 >= tmp158
    tmp184 = tl.full([1], 256, tl.int64)
    tmp185 = tmp0 < tmp184
    tmp186 = tl.load(in_ptr35 + (32*x1 + 96*x2 + 288*x3 + ((-224) + x0)), tmp183, eviction_policy='evict_last', other=0.0)
    tmp187 = tl.load(in_ptr36 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp188 = tmp186 - tmp187
    tmp189 = tl.load(in_ptr37 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp190 = 1e-05
    tmp191 = tmp189 + tmp190
    tmp192 = libdevice.sqrt(tmp191)
    tmp193 = tl.full([1], 1, tl.int32)
    tmp194 = tmp193 / tmp192
    tmp195 = 1.0
    tmp196 = tmp194 * tmp195
    tmp197 = tmp188 * tmp196
    tmp198 = tl.load(in_ptr38 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp199 = tmp197 * tmp198
    tmp200 = tl.load(in_ptr39 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp201 = tmp199 + tmp200
    tmp202 = 0.0
    tmp203 = triton_helpers.maximum(tmp201, tmp202)
    tmp204 = 6.0
    tmp205 = triton_helpers.minimum(tmp203, tmp204)
    tmp206 = tl.full(tmp205.shape, 0.0, tmp205.dtype)
    tmp207 = tl.where(tmp183, tmp205, tmp206)
    tmp208 = tl.where(tmp160, tmp182, tmp207)
    tmp209 = tl.where(tmp134, tmp156, tmp208)
    tmp210 = tl.where(tmp108, tmp130, tmp209)
    tmp211 = tl.where(tmp82, tmp104, tmp210)
    tmp212 = tl.where(tmp56, tmp78, tmp211)
    tmp213 = tl.where(tmp30, tmp52, tmp212)
    tmp214 = tl.where(tmp4, tmp26, tmp213)
    tl.store(out_ptr0 + (x5), tmp214, None)
''', device_str='cuda')


# kernel path: inductor_cache/5e/c5esjqcgp6povp6ubsxk25q7gmwy2i4xptjgo2zgsceafd6h6e7c.py
# Topologically Sorted Source Nodes: [input_160, input_161], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_160 => convolution_55
#   input_161 => add_111, mul_166, mul_167, sub_55
# Graph fragment:
#   %convolution_55 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_5, %primals_332, %primals_333, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_55, %unsqueeze_441), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_445), kwargs = {})
#   %add_111 : [num_users=9] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_447), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sh/cshfnjk5ss45kqbehwmp5st2esjplrwtamowpv7x4wj7ydx6t3jf.py
# Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_162 => convolution_56
# Graph fragment:
#   %convolution_56 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_111, %primals_338, %primals_339, [1, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_50 = async_compile.triton('triton_poi_fused_convolution_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_50(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/no/cno752ke3rpsgpj7fhcwyw4phgxm6temmhvjqfg4jvau4o7p4ylp.py
# Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_174 => convolution_60
# Graph fragment:
#   %convolution_60 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_111, %primals_362, %primals_363, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_51 = async_compile.triton('triton_poi_fused_convolution_51', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_51(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gy/cgy6jy3dyxaccya3ed77daypccxfg7bztcmtse5i6g7u67n36m3m.py
# Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_180 => convolution_62
# Graph fragment:
#   %convolution_62 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_111, %primals_374, %primals_375, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_52 = async_compile.triton('triton_poi_fused_convolution_52', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_52(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5y/c5yq2tdoyhrq6istzafmememreyuiii4mzwsd6u2g7cz3de4oune.py
# Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_6 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_49, %clamp_max_50, %clamp_max_51, %clamp_max_52, %slice_261, %slice_264, %slice_268, %clamp_max_56], 1), kwargs = {})
triton_poi_fused_cat_53 = async_compile.triton('triton_poi_fused_cat_53', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 40, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x4 = xindex // 512
    x3 = xindex // 2048
    x5 = ((xindex // 512) % 4)
    x1 = ((xindex // 512) % 2)
    x6 = xindex // 1024
    x2 = ((xindex // 1024) % 2)
    x7 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = 6.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 128, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr5 + (64*x4 + ((-64) + x0)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr6 + ((-64) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 - tmp32
    tmp34 = tl.load(in_ptr7 + ((-64) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp43 = tl.load(in_ptr8 + ((-64) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-64) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 6.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 192, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = tmp53 & tmp55
    tmp57 = tl.load(in_ptr10 + (64*x4 + ((-128) + x0)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr11 + ((-128) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 - tmp58
    tmp60 = tl.load(in_ptr12 + ((-128) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp61 = 1e-05
    tmp62 = tmp60 + tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = tl.full([1], 1, tl.int32)
    tmp65 = tmp64 / tmp63
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tmp68 = tmp59 * tmp67
    tmp69 = tl.load(in_ptr13 + ((-128) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.load(in_ptr14 + ((-128) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 + tmp71
    tmp73 = 0.0
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = 6.0
    tmp76 = triton_helpers.minimum(tmp74, tmp75)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp56, tmp76, tmp77)
    tmp79 = tmp0 >= tmp54
    tmp80 = tl.full([1], 256, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tmp79 & tmp81
    tmp83 = tl.load(in_ptr15 + (64*x4 + ((-192) + x0)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.load(in_ptr16 + ((-192) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 - tmp84
    tmp86 = tl.load(in_ptr17 + ((-192) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp87 = 1e-05
    tmp88 = tmp86 + tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = tl.full([1], 1, tl.int32)
    tmp91 = tmp90 / tmp89
    tmp92 = 1.0
    tmp93 = tmp91 * tmp92
    tmp94 = tmp85 * tmp93
    tmp95 = tl.load(in_ptr18 + ((-192) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp96 = tmp94 * tmp95
    tmp97 = tl.load(in_ptr19 + ((-192) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp96 + tmp97
    tmp99 = 0.0
    tmp100 = triton_helpers.maximum(tmp98, tmp99)
    tmp101 = 6.0
    tmp102 = triton_helpers.minimum(tmp100, tmp101)
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp82, tmp102, tmp103)
    tmp105 = tmp0 >= tmp80
    tmp106 = tl.full([1], 320, tl.int64)
    tmp107 = tmp0 < tmp106
    tmp108 = tmp105 & tmp107
    tmp109 = tl.load(in_ptr20 + (64*x5 + 384*x3 + ((-256) + x0)), tmp108, eviction_policy='evict_last', other=0.0)
    tmp110 = tl.load(in_ptr21 + ((-256) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp109 - tmp110
    tmp112 = tl.load(in_ptr22 + ((-256) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp113 = 1e-05
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.sqrt(tmp114)
    tmp116 = tl.full([1], 1, tl.int32)
    tmp117 = tmp116 / tmp115
    tmp118 = 1.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp111 * tmp119
    tmp121 = tl.load(in_ptr23 + ((-256) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.load(in_ptr24 + ((-256) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp124 = tmp122 + tmp123
    tmp125 = 0.0
    tmp126 = triton_helpers.maximum(tmp124, tmp125)
    tmp127 = 6.0
    tmp128 = triton_helpers.minimum(tmp126, tmp127)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp108, tmp128, tmp129)
    tmp131 = tmp0 >= tmp106
    tmp132 = tl.full([1], 384, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr25 + (64*x1 + 192*x6 + ((-320) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr26 + ((-320) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp137 = tmp135 - tmp136
    tmp138 = tl.load(in_ptr27 + ((-320) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp139 = 1e-05
    tmp140 = tmp138 + tmp139
    tmp141 = libdevice.sqrt(tmp140)
    tmp142 = tl.full([1], 1, tl.int32)
    tmp143 = tmp142 / tmp141
    tmp144 = 1.0
    tmp145 = tmp143 * tmp144
    tmp146 = tmp137 * tmp145
    tmp147 = tl.load(in_ptr28 + ((-320) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp148 = tmp146 * tmp147
    tmp149 = tl.load(in_ptr29 + ((-320) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp148 + tmp149
    tmp151 = 0.0
    tmp152 = triton_helpers.maximum(tmp150, tmp151)
    tmp153 = 6.0
    tmp154 = triton_helpers.minimum(tmp152, tmp153)
    tmp155 = tl.full(tmp154.shape, 0.0, tmp154.dtype)
    tmp156 = tl.where(tmp134, tmp154, tmp155)
    tmp157 = tmp0 >= tmp132
    tmp158 = tl.full([1], 448, tl.int64)
    tmp159 = tmp0 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tl.load(in_ptr30 + (64*x1 + 192*x2 + 576*x3 + ((-384) + x0)), tmp160, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr31 + ((-384) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp163 = tmp161 - tmp162
    tmp164 = tl.load(in_ptr32 + ((-384) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp165 = 1e-05
    tmp166 = tmp164 + tmp165
    tmp167 = libdevice.sqrt(tmp166)
    tmp168 = tl.full([1], 1, tl.int32)
    tmp169 = tmp168 / tmp167
    tmp170 = 1.0
    tmp171 = tmp169 * tmp170
    tmp172 = tmp163 * tmp171
    tmp173 = tl.load(in_ptr33 + ((-384) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp174 = tmp172 * tmp173
    tmp175 = tl.load(in_ptr34 + ((-384) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp176 = tmp174 + tmp175
    tmp177 = 0.0
    tmp178 = triton_helpers.maximum(tmp176, tmp177)
    tmp179 = 6.0
    tmp180 = triton_helpers.minimum(tmp178, tmp179)
    tmp181 = tl.full(tmp180.shape, 0.0, tmp180.dtype)
    tmp182 = tl.where(tmp160, tmp180, tmp181)
    tmp183 = tmp0 >= tmp158
    tmp184 = tl.full([1], 512, tl.int64)
    tmp185 = tmp0 < tmp184
    tmp186 = tl.load(in_ptr35 + (64*x4 + ((-448) + x0)), tmp183, eviction_policy='evict_last', other=0.0)
    tmp187 = tl.load(in_ptr36 + ((-448) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp188 = tmp186 - tmp187
    tmp189 = tl.load(in_ptr37 + ((-448) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp190 = 1e-05
    tmp191 = tmp189 + tmp190
    tmp192 = libdevice.sqrt(tmp191)
    tmp193 = tl.full([1], 1, tl.int32)
    tmp194 = tmp193 / tmp192
    tmp195 = 1.0
    tmp196 = tmp194 * tmp195
    tmp197 = tmp188 * tmp196
    tmp198 = tl.load(in_ptr38 + ((-448) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp199 = tmp197 * tmp198
    tmp200 = tl.load(in_ptr39 + ((-448) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp201 = tmp199 + tmp200
    tmp202 = 0.0
    tmp203 = triton_helpers.maximum(tmp201, tmp202)
    tmp204 = 6.0
    tmp205 = triton_helpers.minimum(tmp203, tmp204)
    tmp206 = tl.full(tmp205.shape, 0.0, tmp205.dtype)
    tmp207 = tl.where(tmp183, tmp205, tmp206)
    tmp208 = tl.where(tmp160, tmp182, tmp207)
    tmp209 = tl.where(tmp134, tmp156, tmp208)
    tmp210 = tl.where(tmp108, tmp130, tmp209)
    tmp211 = tl.where(tmp82, tmp104, tmp210)
    tmp212 = tl.where(tmp56, tmp78, tmp211)
    tmp213 = tl.where(tmp30, tmp52, tmp212)
    tmp214 = tl.where(tmp4, tmp26, tmp213)
    tl.store(out_ptr0 + (x7), tmp214, None)
''', device_str='cuda')


# kernel path: inductor_cache/st/cstdf3z4z3gkwlweru7xygewdi5iemievyfabplrcxvjdh55mcio.py
# Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_240 => convolution_83
# Graph fragment:
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_165, %primals_500, %primals_501, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_54 = async_compile.triton('triton_poi_fused_convolution_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_54(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yz/cyziowyzk3loer6xdhnnyzyubl77q2qyluwylpjligrmgkx6cvcx.py
# Topologically Sorted Source Nodes: [input_241], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_241 => add_167, mul_250, mul_251, sub_83
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_665), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_250, %unsqueeze_669), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_251, %unsqueeze_671), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 320*x2 + 1280*y1), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 4*y3), tmp15, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (16, ), (1, ))
    assert_size_stride(primals_12, (16, ), (1, ))
    assert_size_stride(primals_13, (16, ), (1, ))
    assert_size_stride(primals_14, (16, 16, 5, 1), (80, 5, 1, 1))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (16, ), (1, ))
    assert_size_stride(primals_17, (16, ), (1, ))
    assert_size_stride(primals_18, (16, ), (1, ))
    assert_size_stride(primals_19, (16, ), (1, ))
    assert_size_stride(primals_20, (16, 16, 1, 5), (80, 5, 5, 1))
    assert_size_stride(primals_21, (16, ), (1, ))
    assert_size_stride(primals_22, (16, ), (1, ))
    assert_size_stride(primals_23, (16, ), (1, ))
    assert_size_stride(primals_24, (16, ), (1, ))
    assert_size_stride(primals_25, (16, ), (1, ))
    assert_size_stride(primals_26, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_27, (16, ), (1, ))
    assert_size_stride(primals_28, (16, ), (1, ))
    assert_size_stride(primals_29, (16, ), (1, ))
    assert_size_stride(primals_30, (16, ), (1, ))
    assert_size_stride(primals_31, (16, ), (1, ))
    assert_size_stride(primals_32, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_33, (16, ), (1, ))
    assert_size_stride(primals_34, (16, ), (1, ))
    assert_size_stride(primals_35, (16, ), (1, ))
    assert_size_stride(primals_36, (16, ), (1, ))
    assert_size_stride(primals_37, (16, ), (1, ))
    assert_size_stride(primals_38, (16, 16, 2, 1), (32, 2, 1, 1))
    assert_size_stride(primals_39, (16, ), (1, ))
    assert_size_stride(primals_40, (16, ), (1, ))
    assert_size_stride(primals_41, (16, ), (1, ))
    assert_size_stride(primals_42, (16, ), (1, ))
    assert_size_stride(primals_43, (16, ), (1, ))
    assert_size_stride(primals_44, (16, 16, 1, 2), (32, 2, 2, 1))
    assert_size_stride(primals_45, (16, ), (1, ))
    assert_size_stride(primals_46, (16, ), (1, ))
    assert_size_stride(primals_47, (16, ), (1, ))
    assert_size_stride(primals_48, (16, ), (1, ))
    assert_size_stride(primals_49, (16, ), (1, ))
    assert_size_stride(primals_50, (16, 16, 2, 2), (64, 4, 2, 1))
    assert_size_stride(primals_51, (16, ), (1, ))
    assert_size_stride(primals_52, (16, ), (1, ))
    assert_size_stride(primals_53, (16, ), (1, ))
    assert_size_stride(primals_54, (16, ), (1, ))
    assert_size_stride(primals_55, (16, ), (1, ))
    assert_size_stride(primals_56, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_57, (16, ), (1, ))
    assert_size_stride(primals_58, (16, ), (1, ))
    assert_size_stride(primals_59, (16, ), (1, ))
    assert_size_stride(primals_60, (16, ), (1, ))
    assert_size_stride(primals_61, (16, ), (1, ))
    assert_size_stride(primals_62, (24, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_63, (24, ), (1, ))
    assert_size_stride(primals_64, (24, ), (1, ))
    assert_size_stride(primals_65, (24, ), (1, ))
    assert_size_stride(primals_66, (24, ), (1, ))
    assert_size_stride(primals_67, (24, ), (1, ))
    assert_size_stride(primals_68, (24, 24, 5, 1), (120, 5, 1, 1))
    assert_size_stride(primals_69, (24, ), (1, ))
    assert_size_stride(primals_70, (24, ), (1, ))
    assert_size_stride(primals_71, (24, ), (1, ))
    assert_size_stride(primals_72, (24, ), (1, ))
    assert_size_stride(primals_73, (24, ), (1, ))
    assert_size_stride(primals_74, (24, 24, 1, 5), (120, 5, 5, 1))
    assert_size_stride(primals_75, (24, ), (1, ))
    assert_size_stride(primals_76, (24, ), (1, ))
    assert_size_stride(primals_77, (24, ), (1, ))
    assert_size_stride(primals_78, (24, ), (1, ))
    assert_size_stride(primals_79, (24, ), (1, ))
    assert_size_stride(primals_80, (24, 24, 3, 1), (72, 3, 1, 1))
    assert_size_stride(primals_81, (24, ), (1, ))
    assert_size_stride(primals_82, (24, ), (1, ))
    assert_size_stride(primals_83, (24, ), (1, ))
    assert_size_stride(primals_84, (24, ), (1, ))
    assert_size_stride(primals_85, (24, ), (1, ))
    assert_size_stride(primals_86, (24, 24, 1, 3), (72, 3, 3, 1))
    assert_size_stride(primals_87, (24, ), (1, ))
    assert_size_stride(primals_88, (24, ), (1, ))
    assert_size_stride(primals_89, (24, ), (1, ))
    assert_size_stride(primals_90, (24, ), (1, ))
    assert_size_stride(primals_91, (24, ), (1, ))
    assert_size_stride(primals_92, (24, 24, 2, 1), (48, 2, 1, 1))
    assert_size_stride(primals_93, (24, ), (1, ))
    assert_size_stride(primals_94, (24, ), (1, ))
    assert_size_stride(primals_95, (24, ), (1, ))
    assert_size_stride(primals_96, (24, ), (1, ))
    assert_size_stride(primals_97, (24, ), (1, ))
    assert_size_stride(primals_98, (24, 24, 1, 2), (48, 2, 2, 1))
    assert_size_stride(primals_99, (24, ), (1, ))
    assert_size_stride(primals_100, (24, ), (1, ))
    assert_size_stride(primals_101, (24, ), (1, ))
    assert_size_stride(primals_102, (24, ), (1, ))
    assert_size_stride(primals_103, (24, ), (1, ))
    assert_size_stride(primals_104, (24, 24, 2, 2), (96, 4, 2, 1))
    assert_size_stride(primals_105, (24, ), (1, ))
    assert_size_stride(primals_106, (24, ), (1, ))
    assert_size_stride(primals_107, (24, ), (1, ))
    assert_size_stride(primals_108, (24, ), (1, ))
    assert_size_stride(primals_109, (24, ), (1, ))
    assert_size_stride(primals_110, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_111, (24, ), (1, ))
    assert_size_stride(primals_112, (24, ), (1, ))
    assert_size_stride(primals_113, (24, ), (1, ))
    assert_size_stride(primals_114, (24, ), (1, ))
    assert_size_stride(primals_115, (24, ), (1, ))
    assert_size_stride(primals_116, (24, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_117, (24, ), (1, ))
    assert_size_stride(primals_118, (24, ), (1, ))
    assert_size_stride(primals_119, (24, ), (1, ))
    assert_size_stride(primals_120, (24, ), (1, ))
    assert_size_stride(primals_121, (24, ), (1, ))
    assert_size_stride(primals_122, (24, 24, 5, 1), (120, 5, 1, 1))
    assert_size_stride(primals_123, (24, ), (1, ))
    assert_size_stride(primals_124, (24, ), (1, ))
    assert_size_stride(primals_125, (24, ), (1, ))
    assert_size_stride(primals_126, (24, ), (1, ))
    assert_size_stride(primals_127, (24, ), (1, ))
    assert_size_stride(primals_128, (24, 24, 1, 5), (120, 5, 5, 1))
    assert_size_stride(primals_129, (24, ), (1, ))
    assert_size_stride(primals_130, (24, ), (1, ))
    assert_size_stride(primals_131, (24, ), (1, ))
    assert_size_stride(primals_132, (24, ), (1, ))
    assert_size_stride(primals_133, (24, ), (1, ))
    assert_size_stride(primals_134, (24, 24, 3, 1), (72, 3, 1, 1))
    assert_size_stride(primals_135, (24, ), (1, ))
    assert_size_stride(primals_136, (24, ), (1, ))
    assert_size_stride(primals_137, (24, ), (1, ))
    assert_size_stride(primals_138, (24, ), (1, ))
    assert_size_stride(primals_139, (24, ), (1, ))
    assert_size_stride(primals_140, (24, 24, 1, 3), (72, 3, 3, 1))
    assert_size_stride(primals_141, (24, ), (1, ))
    assert_size_stride(primals_142, (24, ), (1, ))
    assert_size_stride(primals_143, (24, ), (1, ))
    assert_size_stride(primals_144, (24, ), (1, ))
    assert_size_stride(primals_145, (24, ), (1, ))
    assert_size_stride(primals_146, (24, 24, 2, 1), (48, 2, 1, 1))
    assert_size_stride(primals_147, (24, ), (1, ))
    assert_size_stride(primals_148, (24, ), (1, ))
    assert_size_stride(primals_149, (24, ), (1, ))
    assert_size_stride(primals_150, (24, ), (1, ))
    assert_size_stride(primals_151, (24, ), (1, ))
    assert_size_stride(primals_152, (24, 24, 1, 2), (48, 2, 2, 1))
    assert_size_stride(primals_153, (24, ), (1, ))
    assert_size_stride(primals_154, (24, ), (1, ))
    assert_size_stride(primals_155, (24, ), (1, ))
    assert_size_stride(primals_156, (24, ), (1, ))
    assert_size_stride(primals_157, (24, ), (1, ))
    assert_size_stride(primals_158, (24, 24, 2, 2), (96, 4, 2, 1))
    assert_size_stride(primals_159, (24, ), (1, ))
    assert_size_stride(primals_160, (24, ), (1, ))
    assert_size_stride(primals_161, (24, ), (1, ))
    assert_size_stride(primals_162, (24, ), (1, ))
    assert_size_stride(primals_163, (24, ), (1, ))
    assert_size_stride(primals_164, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_165, (24, ), (1, ))
    assert_size_stride(primals_166, (24, ), (1, ))
    assert_size_stride(primals_167, (24, ), (1, ))
    assert_size_stride(primals_168, (24, ), (1, ))
    assert_size_stride(primals_169, (24, ), (1, ))
    assert_size_stride(primals_170, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_171, (32, ), (1, ))
    assert_size_stride(primals_172, (32, ), (1, ))
    assert_size_stride(primals_173, (32, ), (1, ))
    assert_size_stride(primals_174, (32, ), (1, ))
    assert_size_stride(primals_175, (32, ), (1, ))
    assert_size_stride(primals_176, (32, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_177, (32, ), (1, ))
    assert_size_stride(primals_178, (32, ), (1, ))
    assert_size_stride(primals_179, (32, ), (1, ))
    assert_size_stride(primals_180, (32, ), (1, ))
    assert_size_stride(primals_181, (32, ), (1, ))
    assert_size_stride(primals_182, (32, 32, 1, 5), (160, 5, 5, 1))
    assert_size_stride(primals_183, (32, ), (1, ))
    assert_size_stride(primals_184, (32, ), (1, ))
    assert_size_stride(primals_185, (32, ), (1, ))
    assert_size_stride(primals_186, (32, ), (1, ))
    assert_size_stride(primals_187, (32, ), (1, ))
    assert_size_stride(primals_188, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_189, (32, ), (1, ))
    assert_size_stride(primals_190, (32, ), (1, ))
    assert_size_stride(primals_191, (32, ), (1, ))
    assert_size_stride(primals_192, (32, ), (1, ))
    assert_size_stride(primals_193, (32, ), (1, ))
    assert_size_stride(primals_194, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_195, (32, ), (1, ))
    assert_size_stride(primals_196, (32, ), (1, ))
    assert_size_stride(primals_197, (32, ), (1, ))
    assert_size_stride(primals_198, (32, ), (1, ))
    assert_size_stride(primals_199, (32, ), (1, ))
    assert_size_stride(primals_200, (32, 32, 2, 1), (64, 2, 1, 1))
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (32, ), (1, ))
    assert_size_stride(primals_203, (32, ), (1, ))
    assert_size_stride(primals_204, (32, ), (1, ))
    assert_size_stride(primals_205, (32, ), (1, ))
    assert_size_stride(primals_206, (32, 32, 1, 2), (64, 2, 2, 1))
    assert_size_stride(primals_207, (32, ), (1, ))
    assert_size_stride(primals_208, (32, ), (1, ))
    assert_size_stride(primals_209, (32, ), (1, ))
    assert_size_stride(primals_210, (32, ), (1, ))
    assert_size_stride(primals_211, (32, ), (1, ))
    assert_size_stride(primals_212, (32, 32, 2, 2), (128, 4, 2, 1))
    assert_size_stride(primals_213, (32, ), (1, ))
    assert_size_stride(primals_214, (32, ), (1, ))
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (32, ), (1, ))
    assert_size_stride(primals_217, (32, ), (1, ))
    assert_size_stride(primals_218, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_219, (32, ), (1, ))
    assert_size_stride(primals_220, (32, ), (1, ))
    assert_size_stride(primals_221, (32, ), (1, ))
    assert_size_stride(primals_222, (32, ), (1, ))
    assert_size_stride(primals_223, (32, ), (1, ))
    assert_size_stride(primals_224, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_225, (32, ), (1, ))
    assert_size_stride(primals_226, (32, ), (1, ))
    assert_size_stride(primals_227, (32, ), (1, ))
    assert_size_stride(primals_228, (32, ), (1, ))
    assert_size_stride(primals_229, (32, ), (1, ))
    assert_size_stride(primals_230, (32, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_231, (32, ), (1, ))
    assert_size_stride(primals_232, (32, ), (1, ))
    assert_size_stride(primals_233, (32, ), (1, ))
    assert_size_stride(primals_234, (32, ), (1, ))
    assert_size_stride(primals_235, (32, ), (1, ))
    assert_size_stride(primals_236, (32, 32, 1, 5), (160, 5, 5, 1))
    assert_size_stride(primals_237, (32, ), (1, ))
    assert_size_stride(primals_238, (32, ), (1, ))
    assert_size_stride(primals_239, (32, ), (1, ))
    assert_size_stride(primals_240, (32, ), (1, ))
    assert_size_stride(primals_241, (32, ), (1, ))
    assert_size_stride(primals_242, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_243, (32, ), (1, ))
    assert_size_stride(primals_244, (32, ), (1, ))
    assert_size_stride(primals_245, (32, ), (1, ))
    assert_size_stride(primals_246, (32, ), (1, ))
    assert_size_stride(primals_247, (32, ), (1, ))
    assert_size_stride(primals_248, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_249, (32, ), (1, ))
    assert_size_stride(primals_250, (32, ), (1, ))
    assert_size_stride(primals_251, (32, ), (1, ))
    assert_size_stride(primals_252, (32, ), (1, ))
    assert_size_stride(primals_253, (32, ), (1, ))
    assert_size_stride(primals_254, (32, 32, 2, 1), (64, 2, 1, 1))
    assert_size_stride(primals_255, (32, ), (1, ))
    assert_size_stride(primals_256, (32, ), (1, ))
    assert_size_stride(primals_257, (32, ), (1, ))
    assert_size_stride(primals_258, (32, ), (1, ))
    assert_size_stride(primals_259, (32, ), (1, ))
    assert_size_stride(primals_260, (32, 32, 1, 2), (64, 2, 2, 1))
    assert_size_stride(primals_261, (32, ), (1, ))
    assert_size_stride(primals_262, (32, ), (1, ))
    assert_size_stride(primals_263, (32, ), (1, ))
    assert_size_stride(primals_264, (32, ), (1, ))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, 32, 2, 2), (128, 4, 2, 1))
    assert_size_stride(primals_267, (32, ), (1, ))
    assert_size_stride(primals_268, (32, ), (1, ))
    assert_size_stride(primals_269, (32, ), (1, ))
    assert_size_stride(primals_270, (32, ), (1, ))
    assert_size_stride(primals_271, (32, ), (1, ))
    assert_size_stride(primals_272, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_273, (32, ), (1, ))
    assert_size_stride(primals_274, (32, ), (1, ))
    assert_size_stride(primals_275, (32, ), (1, ))
    assert_size_stride(primals_276, (32, ), (1, ))
    assert_size_stride(primals_277, (32, ), (1, ))
    assert_size_stride(primals_278, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_279, (32, ), (1, ))
    assert_size_stride(primals_280, (32, ), (1, ))
    assert_size_stride(primals_281, (32, ), (1, ))
    assert_size_stride(primals_282, (32, ), (1, ))
    assert_size_stride(primals_283, (32, ), (1, ))
    assert_size_stride(primals_284, (32, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_285, (32, ), (1, ))
    assert_size_stride(primals_286, (32, ), (1, ))
    assert_size_stride(primals_287, (32, ), (1, ))
    assert_size_stride(primals_288, (32, ), (1, ))
    assert_size_stride(primals_289, (32, ), (1, ))
    assert_size_stride(primals_290, (32, 32, 1, 5), (160, 5, 5, 1))
    assert_size_stride(primals_291, (32, ), (1, ))
    assert_size_stride(primals_292, (32, ), (1, ))
    assert_size_stride(primals_293, (32, ), (1, ))
    assert_size_stride(primals_294, (32, ), (1, ))
    assert_size_stride(primals_295, (32, ), (1, ))
    assert_size_stride(primals_296, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_297, (32, ), (1, ))
    assert_size_stride(primals_298, (32, ), (1, ))
    assert_size_stride(primals_299, (32, ), (1, ))
    assert_size_stride(primals_300, (32, ), (1, ))
    assert_size_stride(primals_301, (32, ), (1, ))
    assert_size_stride(primals_302, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_303, (32, ), (1, ))
    assert_size_stride(primals_304, (32, ), (1, ))
    assert_size_stride(primals_305, (32, ), (1, ))
    assert_size_stride(primals_306, (32, ), (1, ))
    assert_size_stride(primals_307, (32, ), (1, ))
    assert_size_stride(primals_308, (32, 32, 2, 1), (64, 2, 1, 1))
    assert_size_stride(primals_309, (32, ), (1, ))
    assert_size_stride(primals_310, (32, ), (1, ))
    assert_size_stride(primals_311, (32, ), (1, ))
    assert_size_stride(primals_312, (32, ), (1, ))
    assert_size_stride(primals_313, (32, ), (1, ))
    assert_size_stride(primals_314, (32, 32, 1, 2), (64, 2, 2, 1))
    assert_size_stride(primals_315, (32, ), (1, ))
    assert_size_stride(primals_316, (32, ), (1, ))
    assert_size_stride(primals_317, (32, ), (1, ))
    assert_size_stride(primals_318, (32, ), (1, ))
    assert_size_stride(primals_319, (32, ), (1, ))
    assert_size_stride(primals_320, (32, 32, 2, 2), (128, 4, 2, 1))
    assert_size_stride(primals_321, (32, ), (1, ))
    assert_size_stride(primals_322, (32, ), (1, ))
    assert_size_stride(primals_323, (32, ), (1, ))
    assert_size_stride(primals_324, (32, ), (1, ))
    assert_size_stride(primals_325, (32, ), (1, ))
    assert_size_stride(primals_326, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_327, (32, ), (1, ))
    assert_size_stride(primals_328, (32, ), (1, ))
    assert_size_stride(primals_329, (32, ), (1, ))
    assert_size_stride(primals_330, (32, ), (1, ))
    assert_size_stride(primals_331, (32, ), (1, ))
    assert_size_stride(primals_332, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_333, (64, ), (1, ))
    assert_size_stride(primals_334, (64, ), (1, ))
    assert_size_stride(primals_335, (64, ), (1, ))
    assert_size_stride(primals_336, (64, ), (1, ))
    assert_size_stride(primals_337, (64, ), (1, ))
    assert_size_stride(primals_338, (64, 64, 5, 1), (320, 5, 1, 1))
    assert_size_stride(primals_339, (64, ), (1, ))
    assert_size_stride(primals_340, (64, ), (1, ))
    assert_size_stride(primals_341, (64, ), (1, ))
    assert_size_stride(primals_342, (64, ), (1, ))
    assert_size_stride(primals_343, (64, ), (1, ))
    assert_size_stride(primals_344, (64, 64, 1, 5), (320, 5, 5, 1))
    assert_size_stride(primals_345, (64, ), (1, ))
    assert_size_stride(primals_346, (64, ), (1, ))
    assert_size_stride(primals_347, (64, ), (1, ))
    assert_size_stride(primals_348, (64, ), (1, ))
    assert_size_stride(primals_349, (64, ), (1, ))
    assert_size_stride(primals_350, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_351, (64, ), (1, ))
    assert_size_stride(primals_352, (64, ), (1, ))
    assert_size_stride(primals_353, (64, ), (1, ))
    assert_size_stride(primals_354, (64, ), (1, ))
    assert_size_stride(primals_355, (64, ), (1, ))
    assert_size_stride(primals_356, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_357, (64, ), (1, ))
    assert_size_stride(primals_358, (64, ), (1, ))
    assert_size_stride(primals_359, (64, ), (1, ))
    assert_size_stride(primals_360, (64, ), (1, ))
    assert_size_stride(primals_361, (64, ), (1, ))
    assert_size_stride(primals_362, (64, 64, 2, 1), (128, 2, 1, 1))
    assert_size_stride(primals_363, (64, ), (1, ))
    assert_size_stride(primals_364, (64, ), (1, ))
    assert_size_stride(primals_365, (64, ), (1, ))
    assert_size_stride(primals_366, (64, ), (1, ))
    assert_size_stride(primals_367, (64, ), (1, ))
    assert_size_stride(primals_368, (64, 64, 1, 2), (128, 2, 2, 1))
    assert_size_stride(primals_369, (64, ), (1, ))
    assert_size_stride(primals_370, (64, ), (1, ))
    assert_size_stride(primals_371, (64, ), (1, ))
    assert_size_stride(primals_372, (64, ), (1, ))
    assert_size_stride(primals_373, (64, ), (1, ))
    assert_size_stride(primals_374, (64, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(primals_375, (64, ), (1, ))
    assert_size_stride(primals_376, (64, ), (1, ))
    assert_size_stride(primals_377, (64, ), (1, ))
    assert_size_stride(primals_378, (64, ), (1, ))
    assert_size_stride(primals_379, (64, ), (1, ))
    assert_size_stride(primals_380, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_381, (64, ), (1, ))
    assert_size_stride(primals_382, (64, ), (1, ))
    assert_size_stride(primals_383, (64, ), (1, ))
    assert_size_stride(primals_384, (64, ), (1, ))
    assert_size_stride(primals_385, (64, ), (1, ))
    assert_size_stride(primals_386, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_387, (64, ), (1, ))
    assert_size_stride(primals_388, (64, ), (1, ))
    assert_size_stride(primals_389, (64, ), (1, ))
    assert_size_stride(primals_390, (64, ), (1, ))
    assert_size_stride(primals_391, (64, ), (1, ))
    assert_size_stride(primals_392, (64, 64, 5, 1), (320, 5, 1, 1))
    assert_size_stride(primals_393, (64, ), (1, ))
    assert_size_stride(primals_394, (64, ), (1, ))
    assert_size_stride(primals_395, (64, ), (1, ))
    assert_size_stride(primals_396, (64, ), (1, ))
    assert_size_stride(primals_397, (64, ), (1, ))
    assert_size_stride(primals_398, (64, 64, 1, 5), (320, 5, 5, 1))
    assert_size_stride(primals_399, (64, ), (1, ))
    assert_size_stride(primals_400, (64, ), (1, ))
    assert_size_stride(primals_401, (64, ), (1, ))
    assert_size_stride(primals_402, (64, ), (1, ))
    assert_size_stride(primals_403, (64, ), (1, ))
    assert_size_stride(primals_404, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_405, (64, ), (1, ))
    assert_size_stride(primals_406, (64, ), (1, ))
    assert_size_stride(primals_407, (64, ), (1, ))
    assert_size_stride(primals_408, (64, ), (1, ))
    assert_size_stride(primals_409, (64, ), (1, ))
    assert_size_stride(primals_410, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_411, (64, ), (1, ))
    assert_size_stride(primals_412, (64, ), (1, ))
    assert_size_stride(primals_413, (64, ), (1, ))
    assert_size_stride(primals_414, (64, ), (1, ))
    assert_size_stride(primals_415, (64, ), (1, ))
    assert_size_stride(primals_416, (64, 64, 2, 1), (128, 2, 1, 1))
    assert_size_stride(primals_417, (64, ), (1, ))
    assert_size_stride(primals_418, (64, ), (1, ))
    assert_size_stride(primals_419, (64, ), (1, ))
    assert_size_stride(primals_420, (64, ), (1, ))
    assert_size_stride(primals_421, (64, ), (1, ))
    assert_size_stride(primals_422, (64, 64, 1, 2), (128, 2, 2, 1))
    assert_size_stride(primals_423, (64, ), (1, ))
    assert_size_stride(primals_424, (64, ), (1, ))
    assert_size_stride(primals_425, (64, ), (1, ))
    assert_size_stride(primals_426, (64, ), (1, ))
    assert_size_stride(primals_427, (64, ), (1, ))
    assert_size_stride(primals_428, (64, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(primals_429, (64, ), (1, ))
    assert_size_stride(primals_430, (64, ), (1, ))
    assert_size_stride(primals_431, (64, ), (1, ))
    assert_size_stride(primals_432, (64, ), (1, ))
    assert_size_stride(primals_433, (64, ), (1, ))
    assert_size_stride(primals_434, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_435, (64, ), (1, ))
    assert_size_stride(primals_436, (64, ), (1, ))
    assert_size_stride(primals_437, (64, ), (1, ))
    assert_size_stride(primals_438, (64, ), (1, ))
    assert_size_stride(primals_439, (64, ), (1, ))
    assert_size_stride(primals_440, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_441, (64, ), (1, ))
    assert_size_stride(primals_442, (64, ), (1, ))
    assert_size_stride(primals_443, (64, ), (1, ))
    assert_size_stride(primals_444, (64, ), (1, ))
    assert_size_stride(primals_445, (64, ), (1, ))
    assert_size_stride(primals_446, (64, 64, 5, 1), (320, 5, 1, 1))
    assert_size_stride(primals_447, (64, ), (1, ))
    assert_size_stride(primals_448, (64, ), (1, ))
    assert_size_stride(primals_449, (64, ), (1, ))
    assert_size_stride(primals_450, (64, ), (1, ))
    assert_size_stride(primals_451, (64, ), (1, ))
    assert_size_stride(primals_452, (64, 64, 1, 5), (320, 5, 5, 1))
    assert_size_stride(primals_453, (64, ), (1, ))
    assert_size_stride(primals_454, (64, ), (1, ))
    assert_size_stride(primals_455, (64, ), (1, ))
    assert_size_stride(primals_456, (64, ), (1, ))
    assert_size_stride(primals_457, (64, ), (1, ))
    assert_size_stride(primals_458, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_459, (64, ), (1, ))
    assert_size_stride(primals_460, (64, ), (1, ))
    assert_size_stride(primals_461, (64, ), (1, ))
    assert_size_stride(primals_462, (64, ), (1, ))
    assert_size_stride(primals_463, (64, ), (1, ))
    assert_size_stride(primals_464, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_465, (64, ), (1, ))
    assert_size_stride(primals_466, (64, ), (1, ))
    assert_size_stride(primals_467, (64, ), (1, ))
    assert_size_stride(primals_468, (64, ), (1, ))
    assert_size_stride(primals_469, (64, ), (1, ))
    assert_size_stride(primals_470, (64, 64, 2, 1), (128, 2, 1, 1))
    assert_size_stride(primals_471, (64, ), (1, ))
    assert_size_stride(primals_472, (64, ), (1, ))
    assert_size_stride(primals_473, (64, ), (1, ))
    assert_size_stride(primals_474, (64, ), (1, ))
    assert_size_stride(primals_475, (64, ), (1, ))
    assert_size_stride(primals_476, (64, 64, 1, 2), (128, 2, 2, 1))
    assert_size_stride(primals_477, (64, ), (1, ))
    assert_size_stride(primals_478, (64, ), (1, ))
    assert_size_stride(primals_479, (64, ), (1, ))
    assert_size_stride(primals_480, (64, ), (1, ))
    assert_size_stride(primals_481, (64, ), (1, ))
    assert_size_stride(primals_482, (64, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(primals_483, (64, ), (1, ))
    assert_size_stride(primals_484, (64, ), (1, ))
    assert_size_stride(primals_485, (64, ), (1, ))
    assert_size_stride(primals_486, (64, ), (1, ))
    assert_size_stride(primals_487, (64, ), (1, ))
    assert_size_stride(primals_488, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_489, (64, ), (1, ))
    assert_size_stride(primals_490, (64, ), (1, ))
    assert_size_stride(primals_491, (64, ), (1, ))
    assert_size_stride(primals_492, (64, ), (1, ))
    assert_size_stride(primals_493, (64, ), (1, ))
    assert_size_stride(primals_494, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_495, (64, ), (1, ))
    assert_size_stride(primals_496, (64, ), (1, ))
    assert_size_stride(primals_497, (64, ), (1, ))
    assert_size_stride(primals_498, (64, ), (1, ))
    assert_size_stride(primals_499, (64, ), (1, ))
    assert_size_stride(primals_500, (320, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_501, (320, ), (1, ))
    assert_size_stride(primals_502, (320, ), (1, ))
    assert_size_stride(primals_503, (320, ), (1, ))
    assert_size_stride(primals_504, (320, ), (1, ))
    assert_size_stride(primals_505, (320, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_3, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_3
        buf2 = empty_strided_cuda((16, 16, 5, 1), (80, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_14, buf2, 256, 5, grid=grid(256, 5), stream=stream0)
        del primals_14
        buf3 = empty_strided_cuda((16, 16, 1, 5), (80, 1, 80, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_20, buf3, 256, 5, grid=grid(256, 5), stream=stream0)
        del primals_20
        buf4 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_26, buf4, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_26
        buf5 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_32, buf5, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_32
        buf6 = empty_strided_cuda((16, 16, 2, 1), (32, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_38, buf6, 256, 2, grid=grid(256, 2), stream=stream0)
        del primals_38
        buf7 = empty_strided_cuda((16, 16, 1, 2), (32, 1, 32, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_44, buf7, 256, 2, grid=grid(256, 2), stream=stream0)
        del primals_44
        buf8 = empty_strided_cuda((16, 16, 2, 2), (64, 1, 32, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_50, buf8, 256, 4, grid=grid(256, 4), stream=stream0)
        del primals_50
        buf9 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_56, buf9, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_56
        buf10 = empty_strided_cuda((24, 24, 5, 1), (120, 1, 24, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_68, buf10, 576, 5, grid=grid(576, 5), stream=stream0)
        del primals_68
        buf11 = empty_strided_cuda((24, 24, 1, 5), (120, 1, 120, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_74, buf11, 576, 5, grid=grid(576, 5), stream=stream0)
        del primals_74
        buf12 = empty_strided_cuda((24, 24, 3, 1), (72, 1, 24, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_80, buf12, 576, 3, grid=grid(576, 3), stream=stream0)
        del primals_80
        buf13 = empty_strided_cuda((24, 24, 1, 3), (72, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_86, buf13, 576, 3, grid=grid(576, 3), stream=stream0)
        del primals_86
        buf14 = empty_strided_cuda((24, 24, 2, 1), (48, 1, 24, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_92, buf14, 576, 2, grid=grid(576, 2), stream=stream0)
        del primals_92
        buf15 = empty_strided_cuda((24, 24, 1, 2), (48, 1, 48, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_98, buf15, 576, 2, grid=grid(576, 2), stream=stream0)
        del primals_98
        buf16 = empty_strided_cuda((24, 24, 2, 2), (96, 1, 48, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_104, buf16, 576, 4, grid=grid(576, 4), stream=stream0)
        del primals_104
        buf17 = empty_strided_cuda((24, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_110, buf17, 576, 9, grid=grid(576, 9), stream=stream0)
        del primals_110
        buf18 = empty_strided_cuda((24, 24, 5, 1), (120, 1, 24, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_122, buf18, 576, 5, grid=grid(576, 5), stream=stream0)
        del primals_122
        buf19 = empty_strided_cuda((24, 24, 1, 5), (120, 1, 120, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_128, buf19, 576, 5, grid=grid(576, 5), stream=stream0)
        del primals_128
        buf20 = empty_strided_cuda((24, 24, 3, 1), (72, 1, 24, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_134, buf20, 576, 3, grid=grid(576, 3), stream=stream0)
        del primals_134
        buf21 = empty_strided_cuda((24, 24, 1, 3), (72, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_140, buf21, 576, 3, grid=grid(576, 3), stream=stream0)
        del primals_140
        buf22 = empty_strided_cuda((24, 24, 2, 1), (48, 1, 24, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_146, buf22, 576, 2, grid=grid(576, 2), stream=stream0)
        del primals_146
        buf23 = empty_strided_cuda((24, 24, 1, 2), (48, 1, 48, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_152, buf23, 576, 2, grid=grid(576, 2), stream=stream0)
        del primals_152
        buf24 = empty_strided_cuda((24, 24, 2, 2), (96, 1, 48, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_158, buf24, 576, 4, grid=grid(576, 4), stream=stream0)
        del primals_158
        buf25 = empty_strided_cuda((24, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_164, buf25, 576, 9, grid=grid(576, 9), stream=stream0)
        del primals_164
        buf26 = empty_strided_cuda((32, 32, 5, 1), (160, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_176, buf26, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_176
        buf27 = empty_strided_cuda((32, 32, 1, 5), (160, 1, 160, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_182, buf27, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_182
        buf28 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_188, buf28, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_188
        buf29 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_194, buf29, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_194
        buf30 = empty_strided_cuda((32, 32, 2, 1), (64, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_200, buf30, 1024, 2, grid=grid(1024, 2), stream=stream0)
        del primals_200
        buf31 = empty_strided_cuda((32, 32, 1, 2), (64, 1, 64, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_206, buf31, 1024, 2, grid=grid(1024, 2), stream=stream0)
        del primals_206
        buf32 = empty_strided_cuda((32, 32, 2, 2), (128, 1, 64, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_212, buf32, 1024, 4, grid=grid(1024, 4), stream=stream0)
        del primals_212
        buf33 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_218, buf33, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_218
        buf34 = empty_strided_cuda((32, 32, 5, 1), (160, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_230, buf34, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_230
        buf35 = empty_strided_cuda((32, 32, 1, 5), (160, 1, 160, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_236, buf35, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_236
        buf36 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_242, buf36, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_242
        buf37 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_248, buf37, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_248
        buf38 = empty_strided_cuda((32, 32, 2, 1), (64, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_254, buf38, 1024, 2, grid=grid(1024, 2), stream=stream0)
        del primals_254
        buf39 = empty_strided_cuda((32, 32, 1, 2), (64, 1, 64, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_260, buf39, 1024, 2, grid=grid(1024, 2), stream=stream0)
        del primals_260
        buf40 = empty_strided_cuda((32, 32, 2, 2), (128, 1, 64, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_266, buf40, 1024, 4, grid=grid(1024, 4), stream=stream0)
        del primals_266
        buf41 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_272, buf41, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_272
        buf42 = empty_strided_cuda((32, 32, 5, 1), (160, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_284, buf42, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_284
        buf43 = empty_strided_cuda((32, 32, 1, 5), (160, 1, 160, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_290, buf43, 1024, 5, grid=grid(1024, 5), stream=stream0)
        del primals_290
        buf44 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_296, buf44, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_296
        buf45 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(primals_302, buf45, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_302
        buf46 = empty_strided_cuda((32, 32, 2, 1), (64, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_308, buf46, 1024, 2, grid=grid(1024, 2), stream=stream0)
        del primals_308
        buf47 = empty_strided_cuda((32, 32, 1, 2), (64, 1, 64, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(primals_314, buf47, 1024, 2, grid=grid(1024, 2), stream=stream0)
        del primals_314
        buf48 = empty_strided_cuda((32, 32, 2, 2), (128, 1, 64, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_320, buf48, 1024, 4, grid=grid(1024, 4), stream=stream0)
        del primals_320
        buf49 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_326, buf49, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_326
        buf50 = empty_strided_cuda((64, 64, 5, 1), (320, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_338, buf50, 4096, 5, grid=grid(4096, 5), stream=stream0)
        del primals_338
        buf51 = empty_strided_cuda((64, 64, 1, 5), (320, 1, 320, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_344, buf51, 4096, 5, grid=grid(4096, 5), stream=stream0)
        del primals_344
        buf52 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_350, buf52, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_350
        buf53 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_356, buf53, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_356
        buf54 = empty_strided_cuda((64, 64, 2, 1), (128, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_19.run(primals_362, buf54, 4096, 2, grid=grid(4096, 2), stream=stream0)
        del primals_362
        buf55 = empty_strided_cuda((64, 64, 1, 2), (128, 1, 128, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_19.run(primals_368, buf55, 4096, 2, grid=grid(4096, 2), stream=stream0)
        del primals_368
        buf56 = empty_strided_cuda((64, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_20.run(primals_374, buf56, 4096, 4, grid=grid(4096, 4), stream=stream0)
        del primals_374
        buf57 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_21.run(primals_380, buf57, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_380
        buf58 = empty_strided_cuda((64, 64, 5, 1), (320, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_392, buf58, 4096, 5, grid=grid(4096, 5), stream=stream0)
        del primals_392
        buf59 = empty_strided_cuda((64, 64, 1, 5), (320, 1, 320, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_398, buf59, 4096, 5, grid=grid(4096, 5), stream=stream0)
        del primals_398
        buf60 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_404, buf60, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_404
        buf61 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_410, buf61, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_410
        buf62 = empty_strided_cuda((64, 64, 2, 1), (128, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_19.run(primals_416, buf62, 4096, 2, grid=grid(4096, 2), stream=stream0)
        del primals_416
        buf63 = empty_strided_cuda((64, 64, 1, 2), (128, 1, 128, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_19.run(primals_422, buf63, 4096, 2, grid=grid(4096, 2), stream=stream0)
        del primals_422
        buf64 = empty_strided_cuda((64, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_20.run(primals_428, buf64, 4096, 4, grid=grid(4096, 4), stream=stream0)
        del primals_428
        buf65 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_21.run(primals_434, buf65, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_434
        buf66 = empty_strided_cuda((64, 64, 5, 1), (320, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_446, buf66, 4096, 5, grid=grid(4096, 5), stream=stream0)
        del primals_446
        buf67 = empty_strided_cuda((64, 64, 1, 5), (320, 1, 320, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_452, buf67, 4096, 5, grid=grid(4096, 5), stream=stream0)
        del primals_452
        buf68 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_458, buf68, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_458
        buf69 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(primals_464, buf69, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_464
        buf70 = empty_strided_cuda((64, 64, 2, 1), (128, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_19.run(primals_470, buf70, 4096, 2, grid=grid(4096, 2), stream=stream0)
        del primals_470
        buf71 = empty_strided_cuda((64, 64, 1, 2), (128, 1, 128, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_19.run(primals_476, buf71, 4096, 2, grid=grid(4096, 2), stream=stream0)
        del primals_476
        buf72 = empty_strided_cuda((64, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_20.run(primals_482, buf72, 4096, 4, grid=grid(4096, 4), stream=stream0)
        del primals_482
        buf73 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_21.run(primals_488, buf73, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_488
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf75 = buf74; del buf74  # reuse
        buf76 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22.run(buf75, primals_2, primals_4, primals_5, primals_6, primals_7, buf76, 131072, grid=grid(131072), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf78 = buf77; del buf77  # reuse
        buf79 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_23.run(buf78, primals_9, primals_10, primals_11, primals_12, primals_13, buf79, 65536, grid=grid(65536), stream=stream0)
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, buf2, stride=(2, 2), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf81, primals_15, 16384, grid=grid(16384), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf79, buf3, stride=(2, 2), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf83, primals_21, 16384, grid=grid(16384), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf79, buf4, stride=(2, 2), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf85, primals_27, 16384, grid=grid(16384), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf79, buf5, stride=(2, 2), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf87, primals_33, 16384, grid=grid(16384), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf79, buf6, stride=(2, 2), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 16, 17, 16), (4352, 1, 256, 16))
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_25.run(buf89, primals_39, 17408, grid=grid(17408), stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf79, buf7, stride=(2, 2), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 16, 16, 17), (4352, 1, 272, 16))
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_25.run(buf91, primals_45, 17408, grid=grid(17408), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf79, buf8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 16, 17, 17), (4624, 1, 272, 16))
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_26.run(buf93, primals_51, 18496, grid=grid(18496), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf79, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf95, primals_57, 16384, grid=grid(16384), stream=stream0)
        del primals_57
        buf96 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_27.run(buf81, primals_16, primals_17, primals_18, primals_19, buf83, primals_22, primals_23, primals_24, primals_25, buf85, primals_28, primals_29, primals_30, primals_31, buf87, primals_34, primals_35, primals_36, primals_37, buf89, primals_40, primals_41, primals_42, primals_43, buf91, primals_46, primals_47, primals_48, primals_49, buf93, primals_52, primals_53, primals_54, primals_55, buf95, primals_58, primals_59, primals_60, primals_61, buf96, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 24, 16, 16), (6144, 1, 384, 24))
        buf98 = buf97; del buf97  # reuse
        buf99 = empty_strided_cuda((4, 24, 16, 16), (6144, 1, 384, 24), torch.float32)
        # Topologically Sorted Source Nodes: [input_30, input_31], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_28.run(buf98, primals_63, primals_64, primals_65, primals_66, primals_67, buf99, 24576, grid=grid(24576), stream=stream0)
        del primals_63
        del primals_67
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, buf10, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 24, 16, 16), (6144, 1, 384, 24))
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf101, primals_69, 24576, grid=grid(24576), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf99, buf11, stride=(1, 1), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 24, 16, 16), (6144, 1, 384, 24))
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf103, primals_75, 24576, grid=grid(24576), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf99, buf12, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 24, 16, 16), (6144, 1, 384, 24))
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf105, primals_81, 24576, grid=grid(24576), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf99, buf13, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 24, 16, 16), (6144, 1, 384, 24))
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf107, primals_87, 24576, grid=grid(24576), stream=stream0)
        del primals_87
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf99, buf14, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 24, 17, 16), (6528, 1, 384, 24))
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_30.run(buf109, primals_93, 26112, grid=grid(26112), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf99, buf15, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 24, 16, 17), (6528, 1, 408, 24))
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_30.run(buf111, primals_99, 26112, grid=grid(26112), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf99, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 24, 17, 17), (6936, 1, 408, 24))
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf113, primals_105, 27744, grid=grid(27744), stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf99, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 24, 16, 16), (6144, 1, 384, 24))
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf115, primals_111, 24576, grid=grid(24576), stream=stream0)
        del primals_111
        buf116 = empty_strided_cuda((4, 192, 16, 16), (49152, 1, 3072, 192), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf101, primals_70, primals_71, primals_72, primals_73, buf103, primals_76, primals_77, primals_78, primals_79, buf105, primals_82, primals_83, primals_84, primals_85, buf107, primals_88, primals_89, primals_90, primals_91, buf109, primals_94, primals_95, primals_96, primals_97, buf111, primals_100, primals_101, primals_102, primals_103, buf113, primals_106, primals_107, primals_108, primals_109, buf115, primals_112, primals_113, primals_114, primals_115, buf116, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 24, 16, 16), (6144, 1, 384, 24))
        buf118 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf118, primals_117, 24576, grid=grid(24576), stream=stream0)
        del primals_117
        buf119 = empty_strided_cuda((4, 24, 17, 17), (6936, 1, 408, 24), torch.float32)
        # Topologically Sorted Source Nodes: [input_57, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_33.run(buf118, primals_118, primals_119, primals_120, primals_121, buf119, 27744, grid=grid(27744), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, buf18, stride=(2, 2), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 24, 9, 9), (1944, 1, 216, 24))
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(buf121, primals_123, 7776, grid=grid(7776), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf119, buf19, stride=(2, 2), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 24, 9, 9), (1944, 1, 216, 24))
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(buf123, primals_129, 7776, grid=grid(7776), stream=stream0)
        del primals_129
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf119, buf20, stride=(2, 2), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 24, 9, 9), (1944, 1, 216, 24))
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(buf125, primals_135, 7776, grid=grid(7776), stream=stream0)
        del primals_135
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf119, buf21, stride=(2, 2), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 24, 9, 9), (1944, 1, 216, 24))
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(buf127, primals_141, 7776, grid=grid(7776), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf119, buf22, stride=(2, 2), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 24, 9, 9), (1944, 1, 216, 24))
        buf129 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(buf129, primals_147, 7776, grid=grid(7776), stream=stream0)
        del primals_147
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf119, buf23, stride=(2, 2), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 24, 9, 9), (1944, 1, 216, 24))
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(buf131, primals_153, 7776, grid=grid(7776), stream=stream0)
        del primals_153
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf119, buf24, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 24, 9, 9), (1944, 1, 216, 24))
        buf133 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(buf133, primals_159, 7776, grid=grid(7776), stream=stream0)
        del primals_159
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf119, buf25, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 24, 9, 9), (1944, 1, 216, 24))
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(buf135, primals_165, 7776, grid=grid(7776), stream=stream0)
        del primals_165
        buf136 = empty_strided_cuda((4, 192, 8, 8), (12288, 1, 1536, 192), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf121, primals_124, primals_125, primals_126, primals_127, buf123, primals_130, primals_131, primals_132, primals_133, buf125, primals_136, primals_137, primals_138, primals_139, buf127, primals_142, primals_143, primals_144, primals_145, buf129, primals_148, primals_149, primals_150, primals_151, buf131, primals_154, primals_155, primals_156, primals_157, buf133, primals_160, primals_161, primals_162, primals_163, buf135, primals_166, primals_167, primals_168, primals_169, buf136, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf138 = buf137; del buf137  # reuse
        buf139 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_82, input_83], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_36.run(buf138, primals_171, primals_172, primals_173, primals_174, primals_175, buf139, 8192, grid=grid(8192), stream=stream0)
        del primals_171
        del primals_175
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, buf26, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf141, primals_177, 8192, grid=grid(8192), stream=stream0)
        del primals_177
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf139, buf27, stride=(1, 1), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf143 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf143, primals_183, 8192, grid=grid(8192), stream=stream0)
        del primals_183
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf139, buf28, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf145 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf145, primals_189, 8192, grid=grid(8192), stream=stream0)
        del primals_189
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf139, buf29, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf147 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf147, primals_195, 8192, grid=grid(8192), stream=stream0)
        del primals_195
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf139, buf30, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 32, 9, 8), (2304, 1, 256, 32))
        buf149 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(buf149, primals_201, 9216, grid=grid(9216), stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf139, buf31, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 32, 8, 9), (2304, 1, 288, 32))
        buf151 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(buf151, primals_207, 9216, grid=grid(9216), stream=stream0)
        del primals_207
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf139, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 32, 9, 9), (2592, 1, 288, 32))
        buf153 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_39.run(buf153, primals_213, 10368, grid=grid(10368), stream=stream0)
        del primals_213
        # Topologically Sorted Source Nodes: [input_105], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf139, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [input_105], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf155, primals_219, 8192, grid=grid(8192), stream=stream0)
        del primals_219
        buf156 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf141, primals_178, primals_179, primals_180, primals_181, buf143, primals_184, primals_185, primals_186, primals_187, buf145, primals_190, primals_191, primals_192, primals_193, buf147, primals_196, primals_197, primals_198, primals_199, buf149, primals_202, primals_203, primals_204, primals_205, buf151, primals_208, primals_209, primals_210, primals_211, buf153, primals_214, primals_215, primals_216, primals_217, buf155, primals_220, primals_221, primals_222, primals_223, buf156, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf158, primals_225, 8192, grid=grid(8192), stream=stream0)
        del primals_225
        buf159 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf160 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf163 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf166 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf169 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf172 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf175 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf178 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        buf181 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_109, input_110, input_113, input_116, input_119, input_122, input_125, input_128, input_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_41.run(buf158, primals_226, primals_227, primals_228, primals_229, buf159, buf160, buf163, buf166, buf169, buf172, buf175, buf178, buf181, 128, 64, grid=grid(128, 64), stream=stream0)
        del primals_229
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, buf34, stride=(2, 2), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 32, 4, 4), (512, 1, 128, 32))
        del buf160
        buf162 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf162, primals_231, 2048, grid=grid(2048), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [input_113], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, buf35, stride=(2, 2), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 32, 4, 4), (512, 1, 128, 32))
        del buf163
        buf165 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [input_113], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf165, primals_237, 2048, grid=grid(2048), stream=stream0)
        del primals_237
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, buf36, stride=(2, 2), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 32, 4, 4), (512, 1, 128, 32))
        del buf166
        buf168 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf168, primals_243, 2048, grid=grid(2048), stream=stream0)
        del primals_243
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, buf37, stride=(2, 2), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 32, 4, 4), (512, 1, 128, 32))
        del buf169
        buf171 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf171, primals_249, 2048, grid=grid(2048), stream=stream0)
        del primals_249
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, buf38, stride=(2, 2), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 32, 5, 4), (640, 1, 128, 32))
        del buf172
        buf174 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_43.run(buf174, primals_255, 2560, grid=grid(2560), stream=stream0)
        del primals_255
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, buf39, stride=(2, 2), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 32, 4, 5), (640, 1, 160, 32))
        buf177 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_43.run(buf177, primals_261, 2560, grid=grid(2560), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [input_128], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, buf40, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 32, 5, 5), (800, 1, 160, 32))
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_128], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_44.run(buf180, primals_267, 3200, grid=grid(3200), stream=stream0)
        del primals_267
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, buf41, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 32, 4, 4), (512, 1, 128, 32))
        buf183 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf183, primals_273, 2048, grid=grid(2048), stream=stream0)
        del primals_273
        buf184 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_45.run(buf162, primals_232, primals_233, primals_234, primals_235, buf165, primals_238, primals_239, primals_240, primals_241, buf168, primals_244, primals_245, primals_246, primals_247, buf171, primals_250, primals_251, primals_252, primals_253, buf174, primals_256, primals_257, primals_258, primals_259, buf177, primals_262, primals_263, primals_264, primals_265, buf180, primals_268, primals_269, primals_270, primals_271, buf183, primals_274, primals_275, primals_276, primals_277, buf184, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_278, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 32, 4, 4), (512, 1, 128, 32))
        buf186 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf186, primals_279, 2048, grid=grid(2048), stream=stream0)
        del primals_279
        buf187 = empty_strided_cuda((4, 32, 5, 5), (800, 1, 160, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_135, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_46.run(buf186, primals_280, primals_281, primals_282, primals_283, buf187, 3200, grid=grid(3200), stream=stream0)
        del primals_283
        # Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, buf42, stride=(2, 2), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 32, 3, 3), (288, 1, 96, 32))
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(buf189, primals_285, 1152, grid=grid(1152), stream=stream0)
        del primals_285
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf187, buf43, stride=(2, 2), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 32, 3, 3), (288, 1, 96, 32))
        buf191 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(buf191, primals_291, 1152, grid=grid(1152), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [input_142], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf187, buf44, stride=(2, 2), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 32, 3, 3), (288, 1, 96, 32))
        buf193 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [input_142], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(buf193, primals_297, 1152, grid=grid(1152), stream=stream0)
        del primals_297
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf187, buf45, stride=(2, 2), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 32, 3, 3), (288, 1, 96, 32))
        buf195 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(buf195, primals_303, 1152, grid=grid(1152), stream=stream0)
        del primals_303
        # Topologically Sorted Source Nodes: [input_148], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf187, buf46, stride=(2, 2), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 32, 3, 3), (288, 1, 96, 32))
        buf197 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [input_148], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(buf197, primals_309, 1152, grid=grid(1152), stream=stream0)
        del primals_309
        # Topologically Sorted Source Nodes: [input_151], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf187, buf47, stride=(2, 2), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 32, 3, 3), (288, 1, 96, 32))
        buf199 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [input_151], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(buf199, primals_315, 1152, grid=grid(1152), stream=stream0)
        del primals_315
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf187, buf48, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 32, 3, 3), (288, 1, 96, 32))
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(buf201, primals_321, 1152, grid=grid(1152), stream=stream0)
        del primals_321
        # Topologically Sorted Source Nodes: [input_157], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf187, buf49, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 32, 3, 3), (288, 1, 96, 32))
        buf203 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [input_157], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(buf203, primals_327, 1152, grid=grid(1152), stream=stream0)
        del primals_327
        buf204 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf189, primals_286, primals_287, primals_288, primals_289, buf191, primals_292, primals_293, primals_294, primals_295, buf193, primals_298, primals_299, primals_300, primals_301, buf195, primals_304, primals_305, primals_306, primals_307, buf197, primals_310, primals_311, primals_312, primals_313, buf199, primals_316, primals_317, primals_318, primals_319, buf201, primals_322, primals_323, primals_324, primals_325, buf203, primals_328, primals_329, primals_330, primals_331, buf204, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_160], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 64, 2, 2), (256, 1, 128, 64))
        buf206 = buf205; del buf205  # reuse
        buf207 = empty_strided_cuda((4, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_160, input_161], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_49.run(buf206, primals_333, primals_334, primals_335, primals_336, primals_337, buf207, 1024, grid=grid(1024), stream=stream0)
        del primals_333
        del primals_337
        # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, buf50, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 64, 2, 2), (256, 1, 128, 64))
        buf209 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf209, primals_339, 1024, grid=grid(1024), stream=stream0)
        del primals_339
        # Topologically Sorted Source Nodes: [input_165], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf207, buf51, stride=(1, 1), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 64, 2, 2), (256, 1, 128, 64))
        buf211 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [input_165], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf211, primals_345, 1024, grid=grid(1024), stream=stream0)
        del primals_345
        # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf207, buf52, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 64, 2, 2), (256, 1, 128, 64))
        buf213 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf213, primals_351, 1024, grid=grid(1024), stream=stream0)
        del primals_351
        # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf207, buf53, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 64, 2, 2), (256, 1, 128, 64))
        buf215 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf215, primals_357, 1024, grid=grid(1024), stream=stream0)
        del primals_357
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf207, buf54, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 64, 3, 2), (384, 1, 128, 64))
        buf217 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_51.run(buf217, primals_363, 1536, grid=grid(1536), stream=stream0)
        del primals_363
        # Topologically Sorted Source Nodes: [input_177], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf207, buf55, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 64, 2, 3), (384, 1, 192, 64))
        buf219 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [input_177], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_51.run(buf219, primals_369, 1536, grid=grid(1536), stream=stream0)
        del primals_369
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf207, buf56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 64, 3, 3), (576, 1, 192, 64))
        buf221 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_52.run(buf221, primals_375, 2304, grid=grid(2304), stream=stream0)
        del primals_375
        # Topologically Sorted Source Nodes: [input_183], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf207, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 64, 2, 2), (256, 1, 128, 64))
        buf223 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [input_183], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf223, primals_381, 1024, grid=grid(1024), stream=stream0)
        del primals_381
        buf224 = reinterpret_tensor(buf181, (4, 512, 2, 2), (2048, 1, 1024, 512), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_53.run(buf209, primals_340, primals_341, primals_342, primals_343, buf211, primals_346, primals_347, primals_348, primals_349, buf213, primals_352, primals_353, primals_354, primals_355, buf215, primals_358, primals_359, primals_360, primals_361, buf217, primals_364, primals_365, primals_366, primals_367, buf219, primals_370, primals_371, primals_372, primals_373, buf221, primals_376, primals_377, primals_378, primals_379, buf223, primals_382, primals_383, primals_384, primals_385, buf224, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_186], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_386, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 64, 2, 2), (256, 1, 128, 64))
        buf226 = buf225; del buf225  # reuse
        buf227 = empty_strided_cuda((4, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_186, input_187], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_49.run(buf226, primals_387, primals_388, primals_389, primals_390, primals_391, buf227, 1024, grid=grid(1024), stream=stream0)
        del primals_387
        del primals_391
        # Topologically Sorted Source Nodes: [input_188], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, buf58, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 64, 2, 2), (256, 1, 128, 64))
        buf229 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [input_188], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf229, primals_393, 1024, grid=grid(1024), stream=stream0)
        del primals_393
        # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf227, buf59, stride=(1, 1), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 64, 2, 2), (256, 1, 128, 64))
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf231, primals_399, 1024, grid=grid(1024), stream=stream0)
        del primals_399
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf227, buf60, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 64, 2, 2), (256, 1, 128, 64))
        buf233 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf233, primals_405, 1024, grid=grid(1024), stream=stream0)
        del primals_405
        # Topologically Sorted Source Nodes: [input_197], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf227, buf61, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 64, 2, 2), (256, 1, 128, 64))
        buf235 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [input_197], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf235, primals_411, 1024, grid=grid(1024), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [input_200], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf227, buf62, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 64, 3, 2), (384, 1, 128, 64))
        buf237 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [input_200], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_51.run(buf237, primals_417, 1536, grid=grid(1536), stream=stream0)
        del primals_417
        # Topologically Sorted Source Nodes: [input_203], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf227, buf63, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 64, 2, 3), (384, 1, 192, 64))
        buf239 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [input_203], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_51.run(buf239, primals_423, 1536, grid=grid(1536), stream=stream0)
        del primals_423
        # Topologically Sorted Source Nodes: [input_206], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf227, buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 64, 3, 3), (576, 1, 192, 64))
        buf241 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [input_206], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_52.run(buf241, primals_429, 2304, grid=grid(2304), stream=stream0)
        del primals_429
        # Topologically Sorted Source Nodes: [input_209], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf227, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 64, 2, 2), (256, 1, 128, 64))
        buf243 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [input_209], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf243, primals_435, 1024, grid=grid(1024), stream=stream0)
        del primals_435
        buf244 = reinterpret_tensor(buf178, (4, 512, 2, 2), (2048, 1, 1024, 512), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_53.run(buf229, primals_394, primals_395, primals_396, primals_397, buf231, primals_400, primals_401, primals_402, primals_403, buf233, primals_406, primals_407, primals_408, primals_409, buf235, primals_412, primals_413, primals_414, primals_415, buf237, primals_418, primals_419, primals_420, primals_421, buf239, primals_424, primals_425, primals_426, primals_427, buf241, primals_430, primals_431, primals_432, primals_433, buf243, primals_436, primals_437, primals_438, primals_439, buf244, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_212], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_440, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 64, 2, 2), (256, 1, 128, 64))
        buf246 = buf245; del buf245  # reuse
        buf247 = empty_strided_cuda((4, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_212, input_213], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_49.run(buf246, primals_441, primals_442, primals_443, primals_444, primals_445, buf247, 1024, grid=grid(1024), stream=stream0)
        del primals_441
        del primals_445
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, buf66, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 64, 2, 2), (256, 1, 128, 64))
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf249, primals_447, 1024, grid=grid(1024), stream=stream0)
        del primals_447
        # Topologically Sorted Source Nodes: [input_217], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf247, buf67, stride=(1, 1), padding=(0, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 64, 2, 2), (256, 1, 128, 64))
        buf251 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [input_217], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf251, primals_453, 1024, grid=grid(1024), stream=stream0)
        del primals_453
        # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf247, buf68, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 64, 2, 2), (256, 1, 128, 64))
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf253, primals_459, 1024, grid=grid(1024), stream=stream0)
        del primals_459
        # Topologically Sorted Source Nodes: [input_223], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf247, buf69, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 64, 2, 2), (256, 1, 128, 64))
        buf255 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [input_223], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf255, primals_465, 1024, grid=grid(1024), stream=stream0)
        del primals_465
        # Topologically Sorted Source Nodes: [input_226], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf247, buf70, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 64, 3, 2), (384, 1, 128, 64))
        buf257 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [input_226], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_51.run(buf257, primals_471, 1536, grid=grid(1536), stream=stream0)
        del primals_471
        # Topologically Sorted Source Nodes: [input_229], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf247, buf71, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 64, 2, 3), (384, 1, 192, 64))
        buf259 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [input_229], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_51.run(buf259, primals_477, 1536, grid=grid(1536), stream=stream0)
        del primals_477
        # Topologically Sorted Source Nodes: [input_232], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf247, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 64, 3, 3), (576, 1, 192, 64))
        buf261 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [input_232], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_52.run(buf261, primals_483, 2304, grid=grid(2304), stream=stream0)
        del primals_483
        # Topologically Sorted Source Nodes: [input_235], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf247, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 64, 2, 2), (256, 1, 128, 64))
        buf263 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [input_235], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf263, primals_489, 1024, grid=grid(1024), stream=stream0)
        del primals_489
        buf264 = reinterpret_tensor(buf175, (4, 512, 2, 2), (2048, 1, 1024, 512), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_53.run(buf249, primals_448, primals_449, primals_450, primals_451, buf251, primals_454, primals_455, primals_456, primals_457, buf253, primals_460, primals_461, primals_462, primals_463, buf255, primals_466, primals_467, primals_468, primals_469, buf257, primals_472, primals_473, primals_474, primals_475, buf259, primals_478, primals_479, primals_480, primals_481, buf261, primals_484, primals_485, primals_486, primals_487, buf263, primals_490, primals_491, primals_492, primals_493, buf264, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_238], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_494, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 64, 2, 2), (256, 1, 128, 64))
        buf266 = buf265; del buf265  # reuse
        buf267 = empty_strided_cuda((4, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_238, input_239], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_49.run(buf266, primals_495, primals_496, primals_497, primals_498, primals_499, buf267, 1024, grid=grid(1024), stream=stream0)
        del primals_495
        del primals_499
        # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_500, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 320, 2, 2), (1280, 1, 640, 320))
        buf269 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_54.run(buf269, primals_501, 5120, grid=grid(5120), stream=stream0)
        del primals_501
        buf270 = empty_strided_cuda((4, 320, 2, 2), (1280, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_241], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_55.run(buf269, primals_502, primals_503, primals_504, primals_505, buf270, 1280, 4, grid=grid(1280, 4), stream=stream0)
        del primals_505
    return (buf159, buf270, buf0, buf1, primals_4, primals_5, primals_6, primals_7, primals_8, primals_10, primals_11, primals_12, buf2, primals_16, primals_17, primals_18, primals_19, buf3, primals_22, primals_23, primals_24, primals_25, buf4, primals_28, primals_29, primals_30, primals_31, buf5, primals_34, primals_35, primals_36, primals_37, buf6, primals_40, primals_41, primals_42, primals_43, buf7, primals_46, primals_47, primals_48, primals_49, buf8, primals_52, primals_53, primals_54, primals_55, buf9, primals_58, primals_59, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, buf10, primals_70, primals_71, primals_72, primals_73, buf11, primals_76, primals_77, primals_78, primals_79, buf12, primals_82, primals_83, primals_84, primals_85, buf13, primals_88, primals_89, primals_90, primals_91, buf14, primals_94, primals_95, primals_96, primals_97, buf15, primals_100, primals_101, primals_102, primals_103, buf16, primals_106, primals_107, primals_108, primals_109, buf17, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_119, primals_120, buf18, primals_124, primals_125, primals_126, primals_127, buf19, primals_130, primals_131, primals_132, primals_133, buf20, primals_136, primals_137, primals_138, primals_139, buf21, primals_142, primals_143, primals_144, primals_145, buf22, primals_148, primals_149, primals_150, primals_151, buf23, primals_154, primals_155, primals_156, primals_157, buf24, primals_160, primals_161, primals_162, primals_163, buf25, primals_166, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, buf26, primals_178, primals_179, primals_180, primals_181, buf27, primals_184, primals_185, primals_186, primals_187, buf28, primals_190, primals_191, primals_192, primals_193, buf29, primals_196, primals_197, primals_198, primals_199, buf30, primals_202, primals_203, primals_204, primals_205, buf31, primals_208, primals_209, primals_210, primals_211, buf32, primals_214, primals_215, primals_216, primals_217, buf33, primals_220, primals_221, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, buf34, primals_232, primals_233, primals_234, primals_235, buf35, primals_238, primals_239, primals_240, primals_241, buf36, primals_244, primals_245, primals_246, primals_247, buf37, primals_250, primals_251, primals_252, primals_253, buf38, primals_256, primals_257, primals_258, primals_259, buf39, primals_262, primals_263, primals_264, primals_265, buf40, primals_268, primals_269, primals_270, primals_271, buf41, primals_274, primals_275, primals_276, primals_277, primals_278, primals_280, primals_281, primals_282, buf42, primals_286, primals_287, primals_288, primals_289, buf43, primals_292, primals_293, primals_294, primals_295, buf44, primals_298, primals_299, primals_300, primals_301, buf45, primals_304, primals_305, primals_306, primals_307, buf46, primals_310, primals_311, primals_312, primals_313, buf47, primals_316, primals_317, primals_318, primals_319, buf48, primals_322, primals_323, primals_324, primals_325, buf49, primals_328, primals_329, primals_330, primals_331, primals_332, primals_334, primals_335, primals_336, buf50, primals_340, primals_341, primals_342, primals_343, buf51, primals_346, primals_347, primals_348, primals_349, buf52, primals_352, primals_353, primals_354, primals_355, buf53, primals_358, primals_359, primals_360, primals_361, buf54, primals_364, primals_365, primals_366, primals_367, buf55, primals_370, primals_371, primals_372, primals_373, buf56, primals_376, primals_377, primals_378, primals_379, buf57, primals_382, primals_383, primals_384, primals_385, primals_386, primals_388, primals_389, primals_390, buf58, primals_394, primals_395, primals_396, primals_397, buf59, primals_400, primals_401, primals_402, primals_403, buf60, primals_406, primals_407, primals_408, primals_409, buf61, primals_412, primals_413, primals_414, primals_415, buf62, primals_418, primals_419, primals_420, primals_421, buf63, primals_424, primals_425, primals_426, primals_427, buf64, primals_430, primals_431, primals_432, primals_433, buf65, primals_436, primals_437, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, buf66, primals_448, primals_449, primals_450, primals_451, buf67, primals_454, primals_455, primals_456, primals_457, buf68, primals_460, primals_461, primals_462, primals_463, buf69, primals_466, primals_467, primals_468, primals_469, buf70, primals_472, primals_473, primals_474, primals_475, buf71, primals_478, primals_479, primals_480, primals_481, buf72, primals_484, primals_485, primals_486, primals_487, buf73, primals_490, primals_491, primals_492, primals_493, primals_494, primals_496, primals_497, primals_498, primals_500, primals_502, primals_503, primals_504, buf75, buf76, buf78, buf79, buf81, buf83, buf85, buf87, buf89, buf91, buf93, buf95, buf96, buf98, buf99, buf101, buf103, buf105, buf107, buf109, buf111, buf113, buf115, buf116, buf118, buf119, buf121, buf123, buf125, buf127, buf129, buf131, buf133, buf135, buf136, buf138, buf139, buf141, buf143, buf145, buf147, buf149, buf151, buf153, buf155, buf156, buf158, buf159, buf162, buf165, buf168, buf171, buf174, buf177, buf180, buf183, buf184, buf186, buf187, buf189, buf191, buf193, buf195, buf197, buf199, buf201, buf203, buf204, buf206, buf207, buf209, buf211, buf213, buf215, buf217, buf219, buf221, buf223, buf224, buf226, buf227, buf229, buf231, buf233, buf235, buf237, buf239, buf241, buf243, buf244, buf246, buf247, buf249, buf251, buf253, buf255, buf257, buf259, buf261, buf263, buf264, buf266, buf267, buf269, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, 16, 5, 1), (80, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, 16, 1, 5), (80, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((16, 16, 2, 1), (32, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((16, 16, 1, 2), (32, 2, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((16, 16, 2, 2), (64, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((24, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((24, 24, 5, 1), (120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((24, 24, 1, 5), (120, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((24, 24, 3, 1), (72, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((24, 24, 1, 3), (72, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((24, 24, 2, 1), (48, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((24, 24, 1, 2), (48, 2, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((24, 24, 2, 2), (96, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((24, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((24, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((24, 24, 5, 1), (120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((24, 24, 1, 5), (120, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((24, 24, 3, 1), (72, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((24, 24, 1, 3), (72, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((24, 24, 2, 1), (48, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((24, 24, 1, 2), (48, 2, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((24, 24, 2, 2), (96, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((24, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((32, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((32, 32, 1, 5), (160, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, 32, 2, 1), (64, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, 32, 1, 2), (64, 2, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((32, 32, 2, 2), (128, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((32, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((32, 32, 1, 5), (160, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((32, 32, 2, 1), (64, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((32, 32, 1, 2), (64, 2, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, 32, 2, 2), (128, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((32, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((32, 32, 1, 5), (160, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((32, 32, 2, 1), (64, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((32, 32, 1, 2), (64, 2, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((32, 32, 2, 2), (128, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((64, 64, 5, 1), (320, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((64, 64, 1, 5), (320, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((64, 64, 2, 1), (128, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((64, 64, 1, 2), (128, 2, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((64, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((64, 64, 5, 1), (320, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((64, 64, 1, 5), (320, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((64, 64, 2, 1), (128, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((64, 64, 1, 2), (128, 2, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((64, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((64, 64, 5, 1), (320, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((64, 64, 1, 5), (320, 5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((64, 64, 2, 1), (128, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((64, 64, 1, 2), (128, 2, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((64, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((320, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
