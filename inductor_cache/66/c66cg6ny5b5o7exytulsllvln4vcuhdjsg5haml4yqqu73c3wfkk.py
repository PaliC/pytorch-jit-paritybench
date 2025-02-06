# AOT ID: ['11_forward']
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


# kernel path: inductor_cache/c4/cc4imz3vggpxflm4k2ligdp7sr7uxvronipdhyb2lyebvnmf6zdy.py
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
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 147*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/oq/coqh2lmz2iujhubeu6j2wbhbnq663ckmbkqalf5q2ssghbq5wccl.py
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
    size_hints={'y': 8192, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h2/ch2ss523eeiiqvpckq6n3yddxgicasrxf4mgdffwogieocds47rs.py
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
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/d3/cd3ljdnskeg7rrcawf67qvhw5avhxeiqv77s76s5ynoydsxshwka.py
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
    size_hints={'y': 32768, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 512*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uu/cuuu3hk5zpvjcwalan2jhkdxjrjfyp6xcju37sl6bjtvf7bxkv6u.py
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
    size_hints={'y': 262144, 'x': 2}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 2
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
    tmp0 = tl.load(in_ptr0 + (x2 + 2*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 1024*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xa/cxaxmvki4krwhlpbmr3rhxahnhup2u7kcc2djwabk2thyhfe2skq.py
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
    size_hints={'y': 131072, 'x': 2}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 2
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
    tmp0 = tl.load(in_ptr0 + (x2 + 2*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 512*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/az/cazwhu53mdp46y55m4qzth22lih7uv7gs74jcqe2yogbn3owmijr.py
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
    ynumel = 16384
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wt/cwtfjygwb5ieddwx7esgosmooxnpnfeihbsbcwgj57xoh42z562u.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/bj/cbjjh25ylchcksiycjcnmtnrsub5nbu2ndtiw77tkg6x2ltjjzbo.py
# Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_4 => convolution_1
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_8, %primals_9, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/bg/cbgyad5lzpa2bndxnc2jwgbkd7ihjicvfrmbbpfvu4z6764zmixg.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x => convolution_3
#   x_1 => add_7, mul_10, mul_11, sub_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_20, %primals_21, [1, 1], [3, 3], [1, 1], False, [0, 0], 128), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-06
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


# kernel path: inductor_cache/cv/ccvwh2iez4q64ltwzzyhbmbb77guyomwi5hc6pisiy6a3n6hex5g.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.convolution, aten.gelu]
# Source node to ATen node mapping:
#   x_2 => convolution_4
#   x_3 => add_8, erf, mul_12, mul_13, mul_14
# Graph fragment:
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_7, %primals_26, %primals_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_4, 0.5), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_4, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_13,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %add_8), kwargs = {})
triton_poi_fused_convolution_gelu_10 = async_compile.triton('triton_poi_fused_convolution_gelu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_10(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/bt/cbtcxw35zyl6k6sq6ru3effz74rjghs7lda2p7ogy7evp6q6cu5l.py
# Topologically Sorted Source Nodes: [x_4, x_5, x_6], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_4 => convolution_5
#   x_5 => mul_15
#   x_6 => add_9
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_14, %primals_28, %primals_29, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_30, %convolution_5), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_2, %mul_15), kwargs = {})
triton_poi_fused_add_convolution_mul_11 = async_compile.triton('triton_poi_fused_add_convolution_mul_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4 * tmp2
    tmp6 = tmp3 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/2o/c2oy7yprs7aibi6yxf7uvyrjwdys2d5dendb7rsjqil3x7mjhnjz.py
# Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_10 => convolution_15
#   input_11 => add_23, mul_38, mul_39, sub_7
#   input_12 => relu_3
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_21, %primals_64, %primals_65, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_57), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %unsqueeze_61), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %unsqueeze_63), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/jh/cjhjckspa3o6zdisb6vc5mmegiwq2gp2i5ivb27atj3d7dpjlddk.py
# Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_28 => convolution_16
#   x_29 => add_25, mul_41, mul_42, sub_8
# Graph fragment:
#   %convolution_16 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %primals_70, %primals_71, [1, 1], [3, 3], [1, 1], False, [0, 0], 256), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_65), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %unsqueeze_69), kwargs = {})
#   %add_25 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %unsqueeze_71), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-06
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


# kernel path: inductor_cache/zd/czd4psgj3vuzkiut2inq2echifcv7f3jux4sw7flhpjnkjn62epf.py
# Topologically Sorted Source Nodes: [x_30, x_31], Original ATen: [aten.convolution, aten.gelu]
# Source node to ATen node mapping:
#   x_30 => convolution_17
#   x_31 => add_26, erf_4, mul_43, mul_44, mul_45
# Graph fragment:
#   %convolution_17 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_25, %primals_76, %primals_77, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_17, 0.5), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_17, 0.7071067811865476), kwargs = {})
#   %erf_4 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_44,), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_45 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %add_26), kwargs = {})
triton_poi_fused_convolution_gelu_14 = async_compile.triton('triton_poi_fused_convolution_gelu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_14(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/pk/cpkkwwhk2gxv7ypddr5mbf576h6th2vutrbrx4ykvk447rp5dz7u.py
# Topologically Sorted Source Nodes: [x_32, x_33, x_34], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_32 => convolution_18
#   x_33 => mul_46
#   x_34 => add_27
# Graph fragment:
#   %convolution_18 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_45, %primals_78, %primals_79, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_80, %convolution_18), kwargs = {})
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_3, %mul_46), kwargs = {})
triton_poi_fused_add_convolution_mul_15 = async_compile.triton('triton_poi_fused_add_convolution_mul_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4 * tmp2
    tmp6 = tmp3 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/23/c23wljx55jy4rhfci3i2qicpnxc2dxp7ayxlplbjjowurhrscllr.py
# Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_13 => convolution_52
#   input_14 => add_73, mul_125, mul_126, sub_20
#   input_15 => relu_4
# Graph fragment:
#   %convolution_52 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_71, %primals_202, %primals_203, [2, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_161), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_125, %unsqueeze_165), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_126, %unsqueeze_167), kwargs = {})
#   %relu_4 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_73,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/og/cogtrl3fbaxtxjvwb5txvhpq4rx5jybrfgwsx4rqm4evfwjggu3m.py
# Topologically Sorted Source Nodes: [x_112, x_113], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_112 => convolution_53
#   x_113 => add_75, mul_128, mul_129, sub_21
# Graph fragment:
#   %convolution_53 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_208, %primals_209, [1, 1], [2, 2], [1, 1], False, [0, 0], 512), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_169), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_128, %unsqueeze_173), kwargs = {})
#   %add_75 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_129, %unsqueeze_175), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-06
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


# kernel path: inductor_cache/64/c64krv4egmqqmtpmpjmvjijbizyqcapl5eul22y6xakii3jhuuvp.py
# Topologically Sorted Source Nodes: [x_114, x_115], Original ATen: [aten.convolution, aten.gelu]
# Source node to ATen node mapping:
#   x_114 => convolution_54
#   x_115 => add_76, erf_16, mul_130, mul_131, mul_132
# Graph fragment:
#   %convolution_54 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %primals_214, %primals_215, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_54, 0.5), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_54, 0.7071067811865476), kwargs = {})
#   %erf_16 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_131,), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_16, 1), kwargs = {})
#   %mul_132 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %add_76), kwargs = {})
triton_poi_fused_convolution_gelu_18 = async_compile.triton('triton_poi_fused_convolution_gelu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_18(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/ha/cha7jcxppk6yusrt7qxdflbi2wumdsyxl2d46jj7osljs2dmiy2h.py
# Topologically Sorted Source Nodes: [x_116, x_117, x_118], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_116 => convolution_55
#   x_117 => mul_133
#   x_118 => add_77
# Graph fragment:
#   %convolution_55 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_132, %primals_216, %primals_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_218, %convolution_55), kwargs = {})
#   %add_77 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_4, %mul_133), kwargs = {})
triton_poi_fused_add_convolution_mul_19 = async_compile.triton('triton_poi_fused_add_convolution_mul_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4 * tmp2
    tmp6 = tmp3 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/zz/czzg5axnhgk44qs2gue76r7w7ybm5i3c44nxoltzbyji3sfh7lxt.py
# Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_16 => convolution_83
#   input_17 => add_115, mul_198, mul_199, sub_31
#   input_18 => relu_5
# Graph fragment:
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_113, %primals_318, %primals_319, [2, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_249), kwargs = {})
#   %mul_198 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_198, %unsqueeze_253), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_199, %unsqueeze_255), kwargs = {})
#   %relu_5 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_115,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/5m/c5m6n73y7vqqraxewxu46jxubeizuh2dacigrjtpg2dyvunly2o4.py
# Topologically Sorted Source Nodes: [x_182, x_183], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_182 => convolution_84
#   x_183 => add_117, mul_201, mul_202, sub_32
# Graph fragment:
#   %convolution_84 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_324, %primals_325, [1, 1], [1, 1], [1, 1], False, [0, 0], 512), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_257), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_201, %unsqueeze_261), kwargs = {})
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_202, %unsqueeze_263), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-06
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


# kernel path: inductor_cache/lz/clzpolstw6frjxjocjwxt55wffcvx43kpm6r3ytunzsxd3xu23qm.py
# Topologically Sorted Source Nodes: [x_184, x_185], Original ATen: [aten.convolution, aten.gelu]
# Source node to ATen node mapping:
#   x_184 => convolution_85
#   x_185 => add_118, erf_26, mul_203, mul_204, mul_205
# Graph fragment:
#   %convolution_85 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_117, %primals_330, %primals_331, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_85, 0.5), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_85, 0.7071067811865476), kwargs = {})
#   %erf_26 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_204,), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_26, 1), kwargs = {})
#   %mul_205 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_203, %add_118), kwargs = {})
triton_poi_fused_convolution_gelu_22 = async_compile.triton('triton_poi_fused_convolution_gelu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_22(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/y2/cy27hgf3eal237elszaix4s3pvaswkkcbf7pvv6fe677pjgznbd5.py
# Topologically Sorted Source Nodes: [x_186, x_187, x_188], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_186 => convolution_86
#   x_187 => mul_206
#   x_188 => add_119
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_205, %primals_332, %primals_333, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_334, %convolution_86), kwargs = {})
#   %add_119 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_5, %mul_206), kwargs = {})
triton_poi_fused_add_convolution_mul_23 = async_compile.triton('triton_poi_fused_add_convolution_mul_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4 * tmp2
    tmp6 = tmp3 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/lq/clqv2hssnemia4wil2a4wrgjlg27mstpcg4zd4ob3j7mk7wumlz6.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_24 = async_compile.triton('triton_poi_fused_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 3
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
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 1536*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/7u/c7u6nkatwtfnynre64zidgoqtvevczedw2sgir5foqypui3ckcbx.py
# Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_19 => convolution_108
# Graph fragment:
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_147, %primals_412, %primals_413, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_25 = async_compile.triton('triton_poi_fused_convolution_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/uf/cufujjxrbpnmdonxyikuonyixtkapicuqgm6ymufpneajk3ahccc.py
# Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_20 => add_149, mul_257, mul_258, sub_40
#   input_21 => relu_6
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_108, %unsqueeze_321), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_257, %unsqueeze_325), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_258, %unsqueeze_327), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_149,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 16384*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x2 + 32*y3), tmp17, xmask)
    tl.store(out_ptr1 + (y0 + 512*x2 + 16384*y1), tmp19, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_31, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_42, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_50, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_61, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_64, (256, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_77, (1024, ), (1, ))
    assert_size_stride(primals_78, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_81, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_82, (256, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_88, (1024, ), (1, ))
    assert_size_stride(primals_89, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, ), (1, ))
    assert_size_stride(primals_98, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_99, (1024, ), (1, ))
    assert_size_stride(primals_100, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_103, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_114, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_115, (256, ), (1, ))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_122, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_125, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (256, ), (1, ))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_135, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_136, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (256, ), (1, ))
    assert_size_stride(primals_139, (256, ), (1, ))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (256, ), (1, ))
    assert_size_stride(primals_142, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_143, (1024, ), (1, ))
    assert_size_stride(primals_144, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_146, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_147, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_148, (256, ), (1, ))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_151, (256, ), (1, ))
    assert_size_stride(primals_152, (256, ), (1, ))
    assert_size_stride(primals_153, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_154, (1024, ), (1, ))
    assert_size_stride(primals_155, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_156, (256, ), (1, ))
    assert_size_stride(primals_157, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_158, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_161, (256, ), (1, ))
    assert_size_stride(primals_162, (256, ), (1, ))
    assert_size_stride(primals_163, (256, ), (1, ))
    assert_size_stride(primals_164, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_165, (1024, ), (1, ))
    assert_size_stride(primals_166, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_167, (256, ), (1, ))
    assert_size_stride(primals_168, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_169, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_170, (256, ), (1, ))
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (256, ), (1, ))
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_176, (1024, ), (1, ))
    assert_size_stride(primals_177, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_178, (256, ), (1, ))
    assert_size_stride(primals_179, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_180, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_181, (256, ), (1, ))
    assert_size_stride(primals_182, (256, ), (1, ))
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_185, (256, ), (1, ))
    assert_size_stride(primals_186, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_187, (1024, ), (1, ))
    assert_size_stride(primals_188, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_190, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_191, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_194, (256, ), (1, ))
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_197, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_198, (1024, ), (1, ))
    assert_size_stride(primals_199, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_202, (512, 256, 2, 1), (512, 2, 1, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_207, (512, ), (1, ))
    assert_size_stride(primals_208, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_209, (512, ), (1, ))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (512, ), (1, ))
    assert_size_stride(primals_212, (512, ), (1, ))
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_215, (2048, ), (1, ))
    assert_size_stride(primals_216, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_217, (512, ), (1, ))
    assert_size_stride(primals_218, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_219, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_220, (512, ), (1, ))
    assert_size_stride(primals_221, (512, ), (1, ))
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (512, ), (1, ))
    assert_size_stride(primals_225, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_226, (2048, ), (1, ))
    assert_size_stride(primals_227, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_228, (512, ), (1, ))
    assert_size_stride(primals_229, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_230, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_233, (512, ), (1, ))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (512, ), (1, ))
    assert_size_stride(primals_236, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_237, (2048, ), (1, ))
    assert_size_stride(primals_238, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_239, (512, ), (1, ))
    assert_size_stride(primals_240, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_241, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_242, (512, ), (1, ))
    assert_size_stride(primals_243, (512, ), (1, ))
    assert_size_stride(primals_244, (512, ), (1, ))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_248, (2048, ), (1, ))
    assert_size_stride(primals_249, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_251, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_252, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_253, (512, ), (1, ))
    assert_size_stride(primals_254, (512, ), (1, ))
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (512, ), (1, ))
    assert_size_stride(primals_258, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_259, (2048, ), (1, ))
    assert_size_stride(primals_260, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_261, (512, ), (1, ))
    assert_size_stride(primals_262, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_263, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (512, ), (1, ))
    assert_size_stride(primals_266, (512, ), (1, ))
    assert_size_stride(primals_267, (512, ), (1, ))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_270, (2048, ), (1, ))
    assert_size_stride(primals_271, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_272, (512, ), (1, ))
    assert_size_stride(primals_273, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_274, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_275, (512, ), (1, ))
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (512, ), (1, ))
    assert_size_stride(primals_278, (512, ), (1, ))
    assert_size_stride(primals_279, (512, ), (1, ))
    assert_size_stride(primals_280, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_281, (2048, ), (1, ))
    assert_size_stride(primals_282, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_283, (512, ), (1, ))
    assert_size_stride(primals_284, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_285, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (512, ), (1, ))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_291, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_292, (2048, ), (1, ))
    assert_size_stride(primals_293, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_296, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_297, (512, ), (1, ))
    assert_size_stride(primals_298, (512, ), (1, ))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_300, (512, ), (1, ))
    assert_size_stride(primals_301, (512, ), (1, ))
    assert_size_stride(primals_302, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_303, (2048, ), (1, ))
    assert_size_stride(primals_304, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_306, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_307, (512, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_308, (512, ), (1, ))
    assert_size_stride(primals_309, (512, ), (1, ))
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, ), (1, ))
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_314, (2048, ), (1, ))
    assert_size_stride(primals_315, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_318, (512, 512, 2, 1), (1024, 2, 1, 1))
    assert_size_stride(primals_319, (512, ), (1, ))
    assert_size_stride(primals_320, (512, ), (1, ))
    assert_size_stride(primals_321, (512, ), (1, ))
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (512, ), (1, ))
    assert_size_stride(primals_324, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_325, (512, ), (1, ))
    assert_size_stride(primals_326, (512, ), (1, ))
    assert_size_stride(primals_327, (512, ), (1, ))
    assert_size_stride(primals_328, (512, ), (1, ))
    assert_size_stride(primals_329, (512, ), (1, ))
    assert_size_stride(primals_330, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_331, (2048, ), (1, ))
    assert_size_stride(primals_332, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_333, (512, ), (1, ))
    assert_size_stride(primals_334, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_335, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_336, (512, ), (1, ))
    assert_size_stride(primals_337, (512, ), (1, ))
    assert_size_stride(primals_338, (512, ), (1, ))
    assert_size_stride(primals_339, (512, ), (1, ))
    assert_size_stride(primals_340, (512, ), (1, ))
    assert_size_stride(primals_341, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_342, (2048, ), (1, ))
    assert_size_stride(primals_343, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_344, (512, ), (1, ))
    assert_size_stride(primals_345, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_346, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_347, (512, ), (1, ))
    assert_size_stride(primals_348, (512, ), (1, ))
    assert_size_stride(primals_349, (512, ), (1, ))
    assert_size_stride(primals_350, (512, ), (1, ))
    assert_size_stride(primals_351, (512, ), (1, ))
    assert_size_stride(primals_352, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_353, (2048, ), (1, ))
    assert_size_stride(primals_354, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_355, (512, ), (1, ))
    assert_size_stride(primals_356, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_357, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_358, (512, ), (1, ))
    assert_size_stride(primals_359, (512, ), (1, ))
    assert_size_stride(primals_360, (512, ), (1, ))
    assert_size_stride(primals_361, (512, ), (1, ))
    assert_size_stride(primals_362, (512, ), (1, ))
    assert_size_stride(primals_363, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_364, (2048, ), (1, ))
    assert_size_stride(primals_365, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_366, (512, ), (1, ))
    assert_size_stride(primals_367, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_368, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_369, (512, ), (1, ))
    assert_size_stride(primals_370, (512, ), (1, ))
    assert_size_stride(primals_371, (512, ), (1, ))
    assert_size_stride(primals_372, (512, ), (1, ))
    assert_size_stride(primals_373, (512, ), (1, ))
    assert_size_stride(primals_374, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_375, (2048, ), (1, ))
    assert_size_stride(primals_376, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_377, (512, ), (1, ))
    assert_size_stride(primals_378, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_379, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_380, (512, ), (1, ))
    assert_size_stride(primals_381, (512, ), (1, ))
    assert_size_stride(primals_382, (512, ), (1, ))
    assert_size_stride(primals_383, (512, ), (1, ))
    assert_size_stride(primals_384, (512, ), (1, ))
    assert_size_stride(primals_385, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_386, (2048, ), (1, ))
    assert_size_stride(primals_387, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_388, (512, ), (1, ))
    assert_size_stride(primals_389, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_390, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_391, (512, ), (1, ))
    assert_size_stride(primals_392, (512, ), (1, ))
    assert_size_stride(primals_393, (512, ), (1, ))
    assert_size_stride(primals_394, (512, ), (1, ))
    assert_size_stride(primals_395, (512, ), (1, ))
    assert_size_stride(primals_396, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_397, (2048, ), (1, ))
    assert_size_stride(primals_398, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_399, (512, ), (1, ))
    assert_size_stride(primals_400, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_401, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_402, (512, ), (1, ))
    assert_size_stride(primals_403, (512, ), (1, ))
    assert_size_stride(primals_404, (512, ), (1, ))
    assert_size_stride(primals_405, (512, ), (1, ))
    assert_size_stride(primals_406, (512, ), (1, ))
    assert_size_stride(primals_407, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_408, (2048, ), (1, ))
    assert_size_stride(primals_409, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_410, (512, ), (1, ))
    assert_size_stride(primals_411, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_412, (512, 512, 3, 1), (1536, 3, 1, 1))
    assert_size_stride(primals_413, (512, ), (1, ))
    assert_size_stride(primals_414, (512, ), (1, ))
    assert_size_stride(primals_415, (512, ), (1, ))
    assert_size_stride(primals_416, (512, ), (1, ))
    assert_size_stride(primals_417, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 49, grid=grid(192, 49), stream=stream0)
        del primals_1
        buf2 = empty_strided_cuda((128, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_8, buf2, 8192, 4, grid=grid(8192, 4), stream=stream0)
        del primals_8
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_3, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_3
        buf4 = empty_strided_cuda((256, 128, 2, 2), (512, 1, 256, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_64, buf4, 32768, 4, grid=grid(32768, 4), stream=stream0)
        del primals_64
        buf6 = empty_strided_cuda((512, 512, 2, 1), (1024, 1, 512, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_318, buf6, 262144, 2, grid=grid(262144, 2), stream=stream0)
        del primals_318
        buf5 = empty_strided_cuda((512, 256, 2, 1), (512, 1, 256, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_202, buf5, 131072, 2, grid=grid(131072, 2), stream=stream0)
        del primals_202
        buf3 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_14, buf3, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_14
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf1, buf0, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((4, 64, 64, 64), (262144, 1, 4096, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf9, primals_2, primals_4, primals_5, primals_6, primals_7, buf10, 1048576, grid=grid(1048576), stream=stream0)
        del primals_2
        del primals_7
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, buf2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf12 = buf11; del buf11  # reuse
        buf13 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf12, primals_9, primals_10, primals_11, primals_12, primals_13, buf13, 524288, grid=grid(524288), stream=stream0)
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf15 = buf14; del buf14  # reuse
        buf16 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf15, primals_15, primals_16, primals_17, primals_18, primals_19, buf16, 524288, grid=grid(524288), stream=stream0)
        del primals_15
        del primals_19
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_20, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf17, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf18 = buf17; del buf17  # reuse
        buf19 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf18, primals_21, primals_22, primals_23, primals_24, primals_25, buf19, 524288, grid=grid(524288), stream=stream0)
        del primals_21
        del primals_25
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 512, 32, 32), (524288, 1, 16384, 512))
        buf21 = buf20; del buf20  # reuse
        buf22 = empty_strided_cuda((4, 512, 32, 32), (524288, 1, 16384, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_10.run(buf21, primals_27, buf22, 2097152, grid=grid(2097152), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5, x_6], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_11.run(buf24, primals_29, buf16, primals_30, buf25, 524288, grid=grid(524288), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_31, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf26, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf27 = buf26; del buf26  # reuse
        buf28 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf27, primals_32, primals_33, primals_34, primals_35, primals_36, buf28, 524288, grid=grid(524288), stream=stream0)
        del primals_32
        del primals_36
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 512, 32, 32), (524288, 1, 16384, 512))
        buf30 = buf29; del buf29  # reuse
        buf31 = empty_strided_cuda((4, 512, 32, 32), (524288, 1, 16384, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, x_10], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_10.run(buf30, primals_38, buf31, 2097152, grid=grid(2097152), stream=stream0)
        del primals_38
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_39, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, x_12, x_13], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_11.run(buf33, primals_40, buf25, primals_41, buf34, 524288, grid=grid(524288), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_42, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf35, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_14, x_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf36, primals_43, primals_44, primals_45, primals_46, primals_47, buf37, 524288, grid=grid(524288), stream=stream0)
        del primals_43
        del primals_47
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_48, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 512, 32, 32), (524288, 1, 16384, 512))
        buf39 = buf38; del buf38  # reuse
        buf40 = empty_strided_cuda((4, 512, 32, 32), (524288, 1, 16384, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_10.run(buf39, primals_49, buf40, 2097152, grid=grid(2097152), stream=stream0)
        del primals_49
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf42 = buf41; del buf41  # reuse
        buf43 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_18, x_19, x_20], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_11.run(buf42, primals_51, buf34, primals_52, buf43, 524288, grid=grid(524288), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_53, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf44, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf45 = buf44; del buf44  # reuse
        buf46 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_21, x_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf45, primals_54, primals_55, primals_56, primals_57, primals_58, buf46, 524288, grid=grid(524288), stream=stream0)
        del primals_54
        del primals_58
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_59, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 512, 32, 32), (524288, 1, 16384, 512))
        buf48 = buf47; del buf47  # reuse
        buf49 = empty_strided_cuda((4, 512, 32, 32), (524288, 1, 16384, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_23, x_24], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_10.run(buf48, primals_60, buf49, 2097152, grid=grid(2097152), stream=stream0)
        del primals_60
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf51 = buf50; del buf50  # reuse
        buf52 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_25, x_26, x_27], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_11.run(buf51, primals_62, buf43, primals_63, buf52, 524288, grid=grid(524288), stream=stream0)
        del primals_62
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, buf4, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf54 = buf53; del buf53  # reuse
        buf55 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(buf54, primals_65, primals_66, primals_67, primals_68, primals_69, buf55, 262144, grid=grid(262144), stream=stream0)
        del primals_65
        del primals_69
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_70, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf56, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf57 = buf56; del buf56  # reuse
        buf58 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf57, primals_71, primals_72, primals_73, primals_74, primals_75, buf58, 262144, grid=grid(262144), stream=stream0)
        del primals_71
        del primals_75
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf60 = buf59; del buf59  # reuse
        buf61 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_30, x_31], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf60, primals_77, buf61, 1048576, grid=grid(1048576), stream=stream0)
        del primals_77
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf63 = buf62; del buf62  # reuse
        buf64 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_32, x_33, x_34], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf63, primals_79, buf55, primals_80, buf64, 262144, grid=grid(262144), stream=stream0)
        del primals_79
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_81, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf65, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf66 = buf65; del buf65  # reuse
        buf67 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_35, x_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf66, primals_82, primals_83, primals_84, primals_85, primals_86, buf67, 262144, grid=grid(262144), stream=stream0)
        del primals_82
        del primals_86
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf69 = buf68; del buf68  # reuse
        buf70 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_37, x_38], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf69, primals_88, buf70, 1048576, grid=grid(1048576), stream=stream0)
        del primals_88
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_89, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf72 = buf71; del buf71  # reuse
        buf73 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_39, x_40, x_41], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf72, primals_90, buf64, primals_91, buf73, 262144, grid=grid(262144), stream=stream0)
        del primals_90
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_92, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf74, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf75 = buf74; del buf74  # reuse
        buf76 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_42, x_43], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf75, primals_93, primals_94, primals_95, primals_96, primals_97, buf76, 262144, grid=grid(262144), stream=stream0)
        del primals_93
        del primals_97
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf78 = buf77; del buf77  # reuse
        buf79 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_44, x_45], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf78, primals_99, buf79, 1048576, grid=grid(1048576), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf81 = buf80; del buf80  # reuse
        buf82 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_46, x_47, x_48], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf81, primals_101, buf73, primals_102, buf82, 262144, grid=grid(262144), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_103, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf83, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf84 = buf83; del buf83  # reuse
        buf85 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf84, primals_104, primals_105, primals_106, primals_107, primals_108, buf85, 262144, grid=grid(262144), stream=stream0)
        del primals_104
        del primals_108
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf87 = buf86; del buf86  # reuse
        buf88 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_51, x_52], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf87, primals_110, buf88, 1048576, grid=grid(1048576), stream=stream0)
        del primals_110
        # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf90 = buf89; del buf89  # reuse
        buf91 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_53, x_54, x_55], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf90, primals_112, buf82, primals_113, buf91, 262144, grid=grid(262144), stream=stream0)
        del primals_112
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_114, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf92, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf93 = buf92; del buf92  # reuse
        buf94 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_56, x_57], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf93, primals_115, primals_116, primals_117, primals_118, primals_119, buf94, 262144, grid=grid(262144), stream=stream0)
        del primals_115
        del primals_119
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf96 = buf95; del buf95  # reuse
        buf97 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_58, x_59], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf96, primals_121, buf97, 1048576, grid=grid(1048576), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf99 = buf98; del buf98  # reuse
        buf100 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_60, x_61, x_62], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf99, primals_123, buf91, primals_124, buf100, 262144, grid=grid(262144), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_125, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf101, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf102 = buf101; del buf101  # reuse
        buf103 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_63, x_64], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf102, primals_126, primals_127, primals_128, primals_129, primals_130, buf103, 262144, grid=grid(262144), stream=stream0)
        del primals_126
        del primals_130
        # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf105 = buf104; del buf104  # reuse
        buf106 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_65, x_66], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf105, primals_132, buf106, 1048576, grid=grid(1048576), stream=stream0)
        del primals_132
        # Topologically Sorted Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf108 = buf107; del buf107  # reuse
        buf109 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_67, x_68, x_69], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf108, primals_134, buf100, primals_135, buf109, 262144, grid=grid(262144), stream=stream0)
        del primals_134
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_136, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf110, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf111 = buf110; del buf110  # reuse
        buf112 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_70, x_71], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf111, primals_137, primals_138, primals_139, primals_140, primals_141, buf112, 262144, grid=grid(262144), stream=stream0)
        del primals_137
        del primals_141
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf114 = buf113; del buf113  # reuse
        buf115 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_72, x_73], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf114, primals_143, buf115, 1048576, grid=grid(1048576), stream=stream0)
        del primals_143
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf117 = buf116; del buf116  # reuse
        buf118 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_74, x_75, x_76], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf117, primals_145, buf109, primals_146, buf118, 262144, grid=grid(262144), stream=stream0)
        del primals_145
        # Topologically Sorted Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_147, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf119, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf120 = buf119; del buf119  # reuse
        buf121 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_77, x_78], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf120, primals_148, primals_149, primals_150, primals_151, primals_152, buf121, 262144, grid=grid(262144), stream=stream0)
        del primals_148
        del primals_152
        # Topologically Sorted Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf123 = buf122; del buf122  # reuse
        buf124 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_79, x_80], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf123, primals_154, buf124, 1048576, grid=grid(1048576), stream=stream0)
        del primals_154
        # Topologically Sorted Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf126 = buf125; del buf125  # reuse
        buf127 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_81, x_82, x_83], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf126, primals_156, buf118, primals_157, buf127, 262144, grid=grid(262144), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_158, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf128, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf129 = buf128; del buf128  # reuse
        buf130 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_84, x_85], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf129, primals_159, primals_160, primals_161, primals_162, primals_163, buf130, 262144, grid=grid(262144), stream=stream0)
        del primals_159
        del primals_163
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf132 = buf131; del buf131  # reuse
        buf133 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_86, x_87], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf132, primals_165, buf133, 1048576, grid=grid(1048576), stream=stream0)
        del primals_165
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf135 = buf134; del buf134  # reuse
        buf136 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_88, x_89, x_90], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf135, primals_167, buf127, primals_168, buf136, 262144, grid=grid(262144), stream=stream0)
        del primals_167
        # Topologically Sorted Source Nodes: [x_91], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_169, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf137, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf138 = buf137; del buf137  # reuse
        buf139 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_91, x_92], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf138, primals_170, primals_171, primals_172, primals_173, primals_174, buf139, 262144, grid=grid(262144), stream=stream0)
        del primals_170
        del primals_174
        # Topologically Sorted Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf141 = buf140; del buf140  # reuse
        buf142 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_93, x_94], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf141, primals_176, buf142, 1048576, grid=grid(1048576), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf144 = buf143; del buf143  # reuse
        buf145 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_95, x_96, x_97], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf144, primals_178, buf136, primals_179, buf145, 262144, grid=grid(262144), stream=stream0)
        del primals_178
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_180, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf146, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf147 = buf146; del buf146  # reuse
        buf148 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_98, x_99], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf147, primals_181, primals_182, primals_183, primals_184, primals_185, buf148, 262144, grid=grid(262144), stream=stream0)
        del primals_181
        del primals_185
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf150 = buf149; del buf149  # reuse
        buf151 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_100, x_101], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf150, primals_187, buf151, 1048576, grid=grid(1048576), stream=stream0)
        del primals_187
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf153 = buf152; del buf152  # reuse
        buf154 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_102, x_103, x_104], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf153, primals_189, buf145, primals_190, buf154, 262144, grid=grid(262144), stream=stream0)
        del primals_189
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_191, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf155, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf156 = buf155; del buf155  # reuse
        buf157 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_105, x_106], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf156, primals_192, primals_193, primals_194, primals_195, primals_196, buf157, 262144, grid=grid(262144), stream=stream0)
        del primals_192
        del primals_196
        # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf159 = buf158; del buf158  # reuse
        buf160 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_107, x_108], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_14.run(buf159, primals_198, buf160, 1048576, grid=grid(1048576), stream=stream0)
        del primals_198
        # Topologically Sorted Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf162 = buf161; del buf161  # reuse
        buf163 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_109, x_110, x_111], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_15.run(buf162, primals_200, buf154, primals_201, buf163, 262144, grid=grid(262144), stream=stream0)
        del primals_200
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, buf5, stride=(2, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf165 = buf164; del buf164  # reuse
        buf166 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf165, primals_203, primals_204, primals_205, primals_206, primals_207, buf166, 262144, grid=grid(262144), stream=stream0)
        del primals_203
        del primals_207
        # Topologically Sorted Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_208, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf167, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf168 = buf167; del buf167  # reuse
        buf169 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_112, x_113], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf168, primals_209, primals_210, primals_211, primals_212, primals_213, buf169, 262144, grid=grid(262144), stream=stream0)
        del primals_209
        del primals_213
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf171 = buf170; del buf170  # reuse
        buf172 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_114, x_115], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf171, primals_215, buf172, 1048576, grid=grid(1048576), stream=stream0)
        del primals_215
        # Topologically Sorted Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf174 = buf173; del buf173  # reuse
        buf175 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_116, x_117, x_118], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf174, primals_217, buf166, primals_218, buf175, 262144, grid=grid(262144), stream=stream0)
        del primals_217
        # Topologically Sorted Source Nodes: [x_119], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_219, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf176, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf177 = buf176; del buf176  # reuse
        buf178 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_119, x_120], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf177, primals_220, primals_221, primals_222, primals_223, primals_224, buf178, 262144, grid=grid(262144), stream=stream0)
        del primals_220
        del primals_224
        # Topologically Sorted Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf180 = buf179; del buf179  # reuse
        buf181 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_121, x_122], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf180, primals_226, buf181, 1048576, grid=grid(1048576), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [x_123], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf183 = buf182; del buf182  # reuse
        buf184 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_123, x_124, x_125], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf183, primals_228, buf175, primals_229, buf184, 262144, grid=grid(262144), stream=stream0)
        del primals_228
        # Topologically Sorted Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_230, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf185, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf186 = buf185; del buf185  # reuse
        buf187 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_126, x_127], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf186, primals_231, primals_232, primals_233, primals_234, primals_235, buf187, 262144, grid=grid(262144), stream=stream0)
        del primals_231
        del primals_235
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf189 = buf188; del buf188  # reuse
        buf190 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_128, x_129], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf189, primals_237, buf190, 1048576, grid=grid(1048576), stream=stream0)
        del primals_237
        # Topologically Sorted Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf192 = buf191; del buf191  # reuse
        buf193 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_130, x_131, x_132], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf192, primals_239, buf184, primals_240, buf193, 262144, grid=grid(262144), stream=stream0)
        del primals_239
        # Topologically Sorted Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_241, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf194, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf195 = buf194; del buf194  # reuse
        buf196 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_133, x_134], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf195, primals_242, primals_243, primals_244, primals_245, primals_246, buf196, 262144, grid=grid(262144), stream=stream0)
        del primals_242
        del primals_246
        # Topologically Sorted Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf198 = buf197; del buf197  # reuse
        buf199 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_135, x_136], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf198, primals_248, buf199, 1048576, grid=grid(1048576), stream=stream0)
        del primals_248
        # Topologically Sorted Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf201 = buf200; del buf200  # reuse
        buf202 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_137, x_138, x_139], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf201, primals_250, buf193, primals_251, buf202, 262144, grid=grid(262144), stream=stream0)
        del primals_250
        # Topologically Sorted Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_252, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf203, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf204 = buf203; del buf203  # reuse
        buf205 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_140, x_141], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf204, primals_253, primals_254, primals_255, primals_256, primals_257, buf205, 262144, grid=grid(262144), stream=stream0)
        del primals_253
        del primals_257
        # Topologically Sorted Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf207 = buf206; del buf206  # reuse
        buf208 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_142, x_143], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf207, primals_259, buf208, 1048576, grid=grid(1048576), stream=stream0)
        del primals_259
        # Topologically Sorted Source Nodes: [x_144], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_260, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf210 = buf209; del buf209  # reuse
        buf211 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_144, x_145, x_146], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf210, primals_261, buf202, primals_262, buf211, 262144, grid=grid(262144), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [x_147], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_263, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf212, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf213 = buf212; del buf212  # reuse
        buf214 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_147, x_148], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf213, primals_264, primals_265, primals_266, primals_267, primals_268, buf214, 262144, grid=grid(262144), stream=stream0)
        del primals_264
        del primals_268
        # Topologically Sorted Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf216 = buf215; del buf215  # reuse
        buf217 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_149, x_150], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf216, primals_270, buf217, 1048576, grid=grid(1048576), stream=stream0)
        del primals_270
        # Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf219 = buf218; del buf218  # reuse
        buf220 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_151, x_152, x_153], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf219, primals_272, buf211, primals_273, buf220, 262144, grid=grid(262144), stream=stream0)
        del primals_272
        # Topologically Sorted Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_274, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf221, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf222 = buf221; del buf221  # reuse
        buf223 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_154, x_155], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf222, primals_275, primals_276, primals_277, primals_278, primals_279, buf223, 262144, grid=grid(262144), stream=stream0)
        del primals_275
        del primals_279
        # Topologically Sorted Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf225 = buf224; del buf224  # reuse
        buf226 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf225, primals_281, buf226, 1048576, grid=grid(1048576), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf228 = buf227; del buf227  # reuse
        buf229 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_158, x_159, x_160], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf228, primals_283, buf220, primals_284, buf229, 262144, grid=grid(262144), stream=stream0)
        del primals_283
        # Topologically Sorted Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_285, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf230, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf231 = buf230; del buf230  # reuse
        buf232 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_161, x_162], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf231, primals_286, primals_287, primals_288, primals_289, primals_290, buf232, 262144, grid=grid(262144), stream=stream0)
        del primals_286
        del primals_290
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf234 = buf233; del buf233  # reuse
        buf235 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_163, x_164], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf234, primals_292, buf235, 1048576, grid=grid(1048576), stream=stream0)
        del primals_292
        # Topologically Sorted Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_293, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf237 = buf236; del buf236  # reuse
        buf238 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_165, x_166, x_167], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf237, primals_294, buf229, primals_295, buf238, 262144, grid=grid(262144), stream=stream0)
        del primals_294
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_296, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf239, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf240 = buf239; del buf239  # reuse
        buf241 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_168, x_169], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf240, primals_297, primals_298, primals_299, primals_300, primals_301, buf241, 262144, grid=grid(262144), stream=stream0)
        del primals_297
        del primals_301
        # Topologically Sorted Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf243 = buf242; del buf242  # reuse
        buf244 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf243, primals_303, buf244, 1048576, grid=grid(1048576), stream=stream0)
        del primals_303
        # Topologically Sorted Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf246 = buf245; del buf245  # reuse
        buf247 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_172, x_173, x_174], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf246, primals_305, buf238, primals_306, buf247, 262144, grid=grid(262144), stream=stream0)
        del primals_305
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_307, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf248, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf249 = buf248; del buf248  # reuse
        buf250 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf249, primals_308, primals_309, primals_310, primals_311, primals_312, buf250, 262144, grid=grid(262144), stream=stream0)
        del primals_308
        del primals_312
        # Topologically Sorted Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_313, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 2048, 8, 16), (262144, 1, 32768, 2048))
        buf252 = buf251; del buf251  # reuse
        buf253 = empty_strided_cuda((4, 2048, 8, 16), (262144, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_177, x_178], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_18.run(buf252, primals_314, buf253, 1048576, grid=grid(1048576), stream=stream0)
        del primals_314
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_315, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 512, 8, 16), (65536, 1, 8192, 512))
        buf255 = buf254; del buf254  # reuse
        buf256 = empty_strided_cuda((4, 512, 8, 16), (65536, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_179, x_180, x_181], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_19.run(buf255, primals_316, buf247, primals_317, buf256, 262144, grid=grid(262144), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, buf6, stride=(2, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf258 = buf257; del buf257  # reuse
        buf259 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf258, primals_319, primals_320, primals_321, primals_322, primals_323, buf259, 131072, grid=grid(131072), stream=stream0)
        del primals_319
        del primals_323
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_324, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf260, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf261 = buf260; del buf260  # reuse
        buf262 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_182, x_183], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_21.run(buf261, primals_325, primals_326, primals_327, primals_328, primals_329, buf262, 131072, grid=grid(131072), stream=stream0)
        del primals_325
        del primals_329
        # Topologically Sorted Source Nodes: [x_184], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_330, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 2048, 4, 16), (131072, 1, 32768, 2048))
        buf264 = buf263; del buf263  # reuse
        buf265 = empty_strided_cuda((4, 2048, 4, 16), (131072, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_184, x_185], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_22.run(buf264, primals_331, buf265, 524288, grid=grid(524288), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [x_186], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf267 = buf266; del buf266  # reuse
        buf268 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_186, x_187, x_188], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_23.run(buf267, primals_333, buf259, primals_334, buf268, 131072, grid=grid(131072), stream=stream0)
        del primals_333
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_335, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf269, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf270 = buf269; del buf269  # reuse
        buf271 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_189, x_190], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_21.run(buf270, primals_336, primals_337, primals_338, primals_339, primals_340, buf271, 131072, grid=grid(131072), stream=stream0)
        del primals_336
        del primals_340
        # Topologically Sorted Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_341, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (4, 2048, 4, 16), (131072, 1, 32768, 2048))
        buf273 = buf272; del buf272  # reuse
        buf274 = empty_strided_cuda((4, 2048, 4, 16), (131072, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_191, x_192], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_22.run(buf273, primals_342, buf274, 524288, grid=grid(524288), stream=stream0)
        del primals_342
        # Topologically Sorted Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_343, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf276 = buf275; del buf275  # reuse
        buf277 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_193, x_194, x_195], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_23.run(buf276, primals_344, buf268, primals_345, buf277, 131072, grid=grid(131072), stream=stream0)
        del primals_344
        # Topologically Sorted Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_346, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf278, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf279 = buf278; del buf278  # reuse
        buf280 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_196, x_197], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_21.run(buf279, primals_347, primals_348, primals_349, primals_350, primals_351, buf280, 131072, grid=grid(131072), stream=stream0)
        del primals_347
        del primals_351
        # Topologically Sorted Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_352, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (4, 2048, 4, 16), (131072, 1, 32768, 2048))
        buf282 = buf281; del buf281  # reuse
        buf283 = empty_strided_cuda((4, 2048, 4, 16), (131072, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_198, x_199], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_22.run(buf282, primals_353, buf283, 524288, grid=grid(524288), stream=stream0)
        del primals_353
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_354, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf285 = buf284; del buf284  # reuse
        buf286 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_200, x_201, x_202], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_23.run(buf285, primals_355, buf277, primals_356, buf286, 131072, grid=grid(131072), stream=stream0)
        del primals_355
        # Topologically Sorted Source Nodes: [x_203], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_357, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf287, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf288 = buf287; del buf287  # reuse
        buf289 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_203, x_204], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_21.run(buf288, primals_358, primals_359, primals_360, primals_361, primals_362, buf289, 131072, grid=grid(131072), stream=stream0)
        del primals_358
        del primals_362
        # Topologically Sorted Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, primals_363, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (4, 2048, 4, 16), (131072, 1, 32768, 2048))
        buf291 = buf290; del buf290  # reuse
        buf292 = empty_strided_cuda((4, 2048, 4, 16), (131072, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_22.run(buf291, primals_364, buf292, 524288, grid=grid(524288), stream=stream0)
        del primals_364
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_365, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf294 = buf293; del buf293  # reuse
        buf295 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_207, x_208, x_209], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_23.run(buf294, primals_366, buf286, primals_367, buf295, 131072, grid=grid(131072), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_368, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf296, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf297 = buf296; del buf296  # reuse
        buf298 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_210, x_211], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_21.run(buf297, primals_369, primals_370, primals_371, primals_372, primals_373, buf298, 131072, grid=grid(131072), stream=stream0)
        del primals_369
        del primals_373
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_374, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 2048, 4, 16), (131072, 1, 32768, 2048))
        buf300 = buf299; del buf299  # reuse
        buf301 = empty_strided_cuda((4, 2048, 4, 16), (131072, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_212, x_213], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_22.run(buf300, primals_375, buf301, 524288, grid=grid(524288), stream=stream0)
        del primals_375
        # Topologically Sorted Source Nodes: [x_214], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_376, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf303 = buf302; del buf302  # reuse
        buf304 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_214, x_215, x_216], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_23.run(buf303, primals_377, buf295, primals_378, buf304, 131072, grid=grid(131072), stream=stream0)
        del primals_377
        # Topologically Sorted Source Nodes: [x_217], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, primals_379, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf305, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf306 = buf305; del buf305  # reuse
        buf307 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_217, x_218], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_21.run(buf306, primals_380, primals_381, primals_382, primals_383, primals_384, buf307, 131072, grid=grid(131072), stream=stream0)
        del primals_380
        del primals_384
        # Topologically Sorted Source Nodes: [x_219], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_385, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 2048, 4, 16), (131072, 1, 32768, 2048))
        buf309 = buf308; del buf308  # reuse
        buf310 = empty_strided_cuda((4, 2048, 4, 16), (131072, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_22.run(buf309, primals_386, buf310, 524288, grid=grid(524288), stream=stream0)
        del primals_386
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf312 = buf311; del buf311  # reuse
        buf313 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_221, x_222, x_223], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_23.run(buf312, primals_388, buf304, primals_389, buf313, 131072, grid=grid(131072), stream=stream0)
        del primals_388
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, primals_390, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf314, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf315 = buf314; del buf314  # reuse
        buf316 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_224, x_225], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_21.run(buf315, primals_391, primals_392, primals_393, primals_394, primals_395, buf316, 131072, grid=grid(131072), stream=stream0)
        del primals_391
        del primals_395
        # Topologically Sorted Source Nodes: [x_226], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_396, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 2048, 4, 16), (131072, 1, 32768, 2048))
        buf318 = buf317; del buf317  # reuse
        buf319 = empty_strided_cuda((4, 2048, 4, 16), (131072, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_226, x_227], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_22.run(buf318, primals_397, buf319, 524288, grid=grid(524288), stream=stream0)
        del primals_397
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_398, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf321 = buf320; del buf320  # reuse
        buf322 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_228, x_229, x_230], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_23.run(buf321, primals_399, buf313, primals_400, buf322, 131072, grid=grid(131072), stream=stream0)
        del primals_399
        # Topologically Sorted Source Nodes: [x_231], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_401, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf323, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf324 = buf323; del buf323  # reuse
        buf325 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_231, x_232], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_21.run(buf324, primals_402, primals_403, primals_404, primals_405, primals_406, buf325, 131072, grid=grid(131072), stream=stream0)
        del primals_402
        del primals_406
        # Topologically Sorted Source Nodes: [x_233], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_407, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (4, 2048, 4, 16), (131072, 1, 32768, 2048))
        buf327 = buf326; del buf326  # reuse
        buf328 = empty_strided_cuda((4, 2048, 4, 16), (131072, 1, 32768, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_233, x_234], Original ATen: [aten.convolution, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_gelu_22.run(buf327, primals_408, buf328, 524288, grid=grid(524288), stream=stream0)
        del primals_408
        # Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_409, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (4, 512, 4, 16), (32768, 1, 8192, 512))
        buf330 = buf329; del buf329  # reuse
        buf331 = empty_strided_cuda((4, 512, 4, 16), (32768, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_235, x_236, x_237], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_23.run(buf330, primals_410, buf322, primals_411, buf331, 131072, grid=grid(131072), stream=stream0)
        del primals_410
        buf7 = empty_strided_cuda((512, 512, 3, 1), (1536, 1, 512, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(primals_412, buf7, 262144, 3, grid=grid(262144, 3), stream=stream0)
        del primals_412
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, buf7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 512, 2, 16), (16384, 1, 8192, 512))
        buf333 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_25.run(buf333, primals_413, 65536, grid=grid(65536), stream=stream0)
        del primals_413
        buf334 = empty_strided_cuda((4, 512, 2, 16), (16384, 32, 16, 1), torch.float32)
        buf335 = empty_strided_cuda((4, 512, 2, 16), (16384, 1, 8192, 512), torch.bool)
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26.run(buf333, primals_414, primals_415, primals_416, primals_417, buf334, buf335, 2048, 32, grid=grid(2048, 32), stream=stream0)
        del primals_417
    return (buf334, buf0, buf1, primals_4, primals_5, primals_6, buf2, primals_10, primals_11, primals_12, buf3, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_28, primals_30, primals_31, primals_33, primals_34, primals_35, primals_37, primals_39, primals_41, primals_42, primals_44, primals_45, primals_46, primals_48, primals_50, primals_52, primals_53, primals_55, primals_56, primals_57, primals_59, primals_61, primals_63, buf4, primals_66, primals_67, primals_68, primals_70, primals_72, primals_73, primals_74, primals_76, primals_78, primals_80, primals_81, primals_83, primals_84, primals_85, primals_87, primals_89, primals_91, primals_92, primals_94, primals_95, primals_96, primals_98, primals_100, primals_102, primals_103, primals_105, primals_106, primals_107, primals_109, primals_111, primals_113, primals_114, primals_116, primals_117, primals_118, primals_120, primals_122, primals_124, primals_125, primals_127, primals_128, primals_129, primals_131, primals_133, primals_135, primals_136, primals_138, primals_139, primals_140, primals_142, primals_144, primals_146, primals_147, primals_149, primals_150, primals_151, primals_153, primals_155, primals_157, primals_158, primals_160, primals_161, primals_162, primals_164, primals_166, primals_168, primals_169, primals_171, primals_172, primals_173, primals_175, primals_177, primals_179, primals_180, primals_182, primals_183, primals_184, primals_186, primals_188, primals_190, primals_191, primals_193, primals_194, primals_195, primals_197, primals_199, primals_201, buf5, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, primals_212, primals_214, primals_216, primals_218, primals_219, primals_221, primals_222, primals_223, primals_225, primals_227, primals_229, primals_230, primals_232, primals_233, primals_234, primals_236, primals_238, primals_240, primals_241, primals_243, primals_244, primals_245, primals_247, primals_249, primals_251, primals_252, primals_254, primals_255, primals_256, primals_258, primals_260, primals_262, primals_263, primals_265, primals_266, primals_267, primals_269, primals_271, primals_273, primals_274, primals_276, primals_277, primals_278, primals_280, primals_282, primals_284, primals_285, primals_287, primals_288, primals_289, primals_291, primals_293, primals_295, primals_296, primals_298, primals_299, primals_300, primals_302, primals_304, primals_306, primals_307, primals_309, primals_310, primals_311, primals_313, primals_315, primals_317, buf6, primals_320, primals_321, primals_322, primals_324, primals_326, primals_327, primals_328, primals_330, primals_332, primals_334, primals_335, primals_337, primals_338, primals_339, primals_341, primals_343, primals_345, primals_346, primals_348, primals_349, primals_350, primals_352, primals_354, primals_356, primals_357, primals_359, primals_360, primals_361, primals_363, primals_365, primals_367, primals_368, primals_370, primals_371, primals_372, primals_374, primals_376, primals_378, primals_379, primals_381, primals_382, primals_383, primals_385, primals_387, primals_389, primals_390, primals_392, primals_393, primals_394, primals_396, primals_398, primals_400, primals_401, primals_403, primals_404, primals_405, primals_407, primals_409, primals_411, buf7, primals_414, primals_415, primals_416, buf9, buf10, buf12, buf13, buf15, buf16, buf18, buf19, buf21, buf22, buf24, buf25, buf27, buf28, buf30, buf31, buf33, buf34, buf36, buf37, buf39, buf40, buf42, buf43, buf45, buf46, buf48, buf49, buf51, buf52, buf54, buf55, buf57, buf58, buf60, buf61, buf63, buf64, buf66, buf67, buf69, buf70, buf72, buf73, buf75, buf76, buf78, buf79, buf81, buf82, buf84, buf85, buf87, buf88, buf90, buf91, buf93, buf94, buf96, buf97, buf99, buf100, buf102, buf103, buf105, buf106, buf108, buf109, buf111, buf112, buf114, buf115, buf117, buf118, buf120, buf121, buf123, buf124, buf126, buf127, buf129, buf130, buf132, buf133, buf135, buf136, buf138, buf139, buf141, buf142, buf144, buf145, buf147, buf148, buf150, buf151, buf153, buf154, buf156, buf157, buf159, buf160, buf162, buf163, buf165, buf166, buf168, buf169, buf171, buf172, buf174, buf175, buf177, buf178, buf180, buf181, buf183, buf184, buf186, buf187, buf189, buf190, buf192, buf193, buf195, buf196, buf198, buf199, buf201, buf202, buf204, buf205, buf207, buf208, buf210, buf211, buf213, buf214, buf216, buf217, buf219, buf220, buf222, buf223, buf225, buf226, buf228, buf229, buf231, buf232, buf234, buf235, buf237, buf238, buf240, buf241, buf243, buf244, buf246, buf247, buf249, buf250, buf252, buf253, buf255, buf256, buf258, buf259, buf261, buf262, buf264, buf265, buf267, buf268, buf270, buf271, buf273, buf274, buf276, buf277, buf279, buf280, buf282, buf283, buf285, buf286, buf288, buf289, buf291, buf292, buf294, buf295, buf297, buf298, buf300, buf301, buf303, buf304, buf306, buf307, buf309, buf310, buf312, buf313, buf315, buf316, buf318, buf319, buf321, buf322, buf324, buf325, buf327, buf328, buf330, buf331, buf333, buf335, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, 256, 2, 1), (512, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((512, 512, 2, 1), (1024, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((512, 512, 3, 1), (1536, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
