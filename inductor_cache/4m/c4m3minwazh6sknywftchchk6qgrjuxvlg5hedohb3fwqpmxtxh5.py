# AOT ID: ['5_forward']
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


# kernel path: inductor_cache/4b/c4bb2ducsq52gv3m3acmalwooc6vqen6tpzeuohwyy53bzmwgr6a.py
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
    size_hints={'y': 1024, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 75*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ln/cln2jbkggh23rxghf3idkx3ewkstdzqb44lqnz3y2ix3qxraxsm6.py
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
    size_hints={'y': 65536, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 4800*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ie/ciet3pwgew2ml7u3xljnmumgapc22umiby42ucioo5v7g2val3hd.py
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
    size_hints={'y': 131072, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 25
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 4800*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/vl/cvlxggdicqlhjg3rw3ue2sbsdmupzwppgowzjxl3wm6hvrnt72xe.py
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
    size_hints={'y': 262144, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147456
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 9600*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xq/cxqi54ycvtn3bkubdru3kqdalsb2usrt6vqxqmxnujlullo27glq.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/n3/cn3b5yey3wsfptftlpnxcnap7bzqiaex2avgiudivk4ytczhvddq.py
# Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_6 = async_compile.triton('triton_per_fused_native_group_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_6(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 64)
    x2 = xindex // 6144
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 192*(((r3 + 128*x1) % 4096)) + 786432*x2 + ((r3 + 128*x1) // 4096)), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp13, None)
    tl.store(out_ptr2 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/54/c54fzxj7swvku7vnm6r2xjruzl7fudj4kkriixc6pe3g5v5qtei5.py
# Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_7 = async_compile.triton('triton_per_fused_native_group_norm_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 96)
    x1 = xindex // 96
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*r2 + 6144*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 96*r2 + 6144*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 96*r2 + 6144*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pf/cpfjsrnuioze7l5frhtnwie7zo4tcaxrcuskacnnluhwwj2xoo2f.py
# Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_per_fused_native_group_norm_8 = async_compile.triton('triton_per_fused_native_group_norm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 16*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 16*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 131072.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/o5/co5gzfldp6w3fp3qgic7xe5hjufkjjs3awlytnw7gpah2l4mbgwf.py
# Topologically Sorted Source Nodes: [group_norm, relu], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm => add_1, mul_1
#   relu => relu
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_2), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused_native_group_norm_relu_9 = async_compile.triton('triton_poi_fused_native_group_norm_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 192)
    x2 = xindex // 786432
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (6*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (6*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/lw/clwpijrbnkan7stqjkwdgmoctxhwe5sxj667uaug4em6spziegp6.py
# Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_2 => var_mean_2
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_10 = async_compile.triton('triton_per_fused_native_group_norm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_10(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 64)
    x2 = xindex // 6144
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 192*(((r3 + 128*x1) % 4096)) + 786432*x2 + ((r3 + 128*x1) // 4096)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2*x0 + 192*(((r3 + 128*x1) % 4096)) + 786432*x2 + ((r3 + 128*x1) // 4096)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, None)
    tl.store(out_ptr1 + (x4), tmp15, None)
    tl.store(out_ptr2 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/j6/cj6lepg75wasgse4glx6zxmwymloy3ean53uvgokkshicxobjus7.py
# Topologically Sorted Source Nodes: [group_norm_2, relu_2], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_2 => add_6, mul_5
#   relu_2 => relu_2
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze_17), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_14), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
triton_poi_fused_native_group_norm_relu_11 = async_compile.triton('triton_poi_fused_native_group_norm_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 192)
    x2 = xindex // 786432
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (6*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (6*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 131072.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/cz/ccz6oy6t745rgot3bzdg44v6cou4ojoefeh44alwnqksca2ueqsn.py
# Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out_1 => add_4
#   out_2 => add_9
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_4), kwargs = {})
triton_poi_fused_add_12 = async_compile.triton('triton_poi_fused_add_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_12(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/zn/cznikuwulc42lo3klzxf2yzij54mvvnjv3q5xlvd7ckfmxm6wpsu.py
# Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out_3 => add_14
# Graph fragment:
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convolution_6), kwargs = {})
triton_poi_fused_add_13 = async_compile.triton('triton_poi_fused_add_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ad/cad6wjv47o33ud3ihu6tfz7trywu2xqg3fofdfjzbz2oseojb44a.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=3] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_14, [2, 2], [2, 2]), kwargs = {})
triton_poi_fused_avg_pool2d_14 = async_compile.triton('triton_poi_fused_avg_pool2d_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 32)
    x2 = xindex // 6144
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1 + 24576*x2), None)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + 384*x1 + 24576*x2), None)
    tmp3 = tl.load(in_ptr0 + (12288 + x0 + 384*x1 + 24576*x2), None)
    tmp5 = tl.load(in_ptr0 + (12480 + x0 + 384*x1 + 24576*x2), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/2f/c2fehpvvf6u63s75nmvqnhcdcr22t5vzknoquaokz6usuv23534o.py
# Topologically Sorted Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_6 => var_mean_6
# Graph fragment:
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_15 = async_compile.triton('triton_per_fused_native_group_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_15(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % 24)
    x1 = ((xindex // 24) % 64)
    x2 = xindex // 1536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (8*x0 + 192*(((r3 + 128*x1) % 1024)) + 196608*x2 + ((r3 + 128*x1) // 1024)), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
    tl.store(out_ptr1 + (x4), tmp16, xmask)
    tl.store(out_ptr2 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3z/c3zxwnyaatyutsmpd7jyey6oytpma3xdhgbahmp2zjsxjjhy5fmd.py
# Topologically Sorted Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_6 => var_mean_6
# Graph fragment:
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_16 = async_compile.triton('triton_per_fused_native_group_norm_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 24)
    x1 = xindex // 24
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 24*r2 + 1536*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 24*r2 + 1536*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 24*r2 + 1536*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kv/ckv54aqufjnt75wdwcpjm7i2encgdqgij3ypgnwkpxva3lempwlf.py
# Topologically Sorted Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_6 => add_15, rsqrt_6, var_mean_6
# Graph fragment:
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_15,), kwargs = {})
triton_per_fused_native_group_norm_17 = async_compile.triton('triton_per_fused_native_group_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 4*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 4*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 32768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m6/cm6o27besqpbhejn62e3b5t3jq6c5fg3s2bp4uri6qftixhn6yiu.py
# Topologically Sorted Source Nodes: [group_norm_6, relu_6], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_6 => add_16, mul_13
#   relu_6 => relu_6
# Graph fragment:
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %unsqueeze_41), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_38), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
triton_poi_fused_native_group_norm_relu_18 = async_compile.triton('triton_poi_fused_native_group_norm_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 192)
    x2 = xindex // 196608
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (6*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (6*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/v3/cv365lmmoagku2oh6aqog7jtnivcanue35fyb3ljo2fwvm5ru6iy.py
# Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out_4 => add_19
# Graph fragment:
#   %add_19 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_9), kwargs = {})
triton_poi_fused_add_19 = async_compile.triton('triton_poi_fused_add_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/b5/cb5hsmlnd4x7ghe4mx4ltc4yx6qud27lvx3qyygmn4tnptoht5ha.py
# Topologically Sorted Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_8 => var_mean_8
# Graph fragment:
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_16, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_20 = async_compile.triton('triton_per_fused_native_group_norm_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_20(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % 48)
    x1 = ((xindex // 48) % 64)
    x2 = xindex // 3072
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (8*x0 + 384*(((r3 + 128*x1) % 1024)) + 393216*x2 + ((r3 + 128*x1) // 1024)), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp13, None)
    tl.store(out_ptr2 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/4i/c4ixtlfxbets2tw55dyul3t3vezk74w3mgpmmgdfkykt2ryuzbzu.py
# Topologically Sorted Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_8 => var_mean_8
# Graph fragment:
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_16, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_21 = async_compile.triton('triton_per_fused_native_group_norm_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 48)
    x1 = xindex // 48
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 48*r2 + 3072*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 48*r2 + 3072*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 48*r2 + 3072*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xe/cxeoz5boktr4gmm7oazec2agsdt4g6aba2ffxfzw6a3243y3zxu5.py
# Topologically Sorted Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_8 => add_20, rsqrt_8, var_mean_8
# Graph fragment:
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_16, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_20,), kwargs = {})
triton_per_fused_native_group_norm_22 = async_compile.triton('triton_per_fused_native_group_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 4*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 4*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 32768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qb/cqbbngpmmu2ktlg47u4fulkai2zr5enjtf2sj7pwlpwjj5wn7ixu.py
# Topologically Sorted Source Nodes: [group_norm_8, relu_8], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_8 => add_21, mul_17
#   relu_8 => relu_8
# Graph fragment:
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %unsqueeze_53), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_50), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
triton_poi_fused_native_group_norm_relu_23 = async_compile.triton('triton_poi_fused_native_group_norm_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 384)
    x2 = xindex // 393216
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (12*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (12*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/yk/cykwvva33ehvozarsxl3f5e73xjfpswa663tzluzlifavcdnwcgr.py
# Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out_5 => add_24
# Graph fragment:
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %convolution_11), kwargs = {})
triton_poi_fused_add_24 = async_compile.triton('triton_poi_fused_add_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/b3/cb3uykyul2s2lbbppxjjhlytilovyw2sxytzwucdpmwonugvks4o.py
# Topologically Sorted Source Nodes: [out_6, out_7], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   out_6 => add_29
#   out_7 => mean
# Graph fragment:
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %convolution_13), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_29, [-1, -2], True), kwargs = {})
triton_red_fused_add_mean_25 = async_compile.triton('triton_red_fused_add_mean_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mean_25(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 384)
    x1 = xindex // 384
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 384*r2 + 49152*x1), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 384*r2 + 49152*x1), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/w7/cw7rws4beixuukodqitfwfml6tpwa7b6lz4mslsie5jfv5dhlimi.py
# Topologically Sorted Source Nodes: [out_6, out_7, relu_12], Original ATen: [aten.add, aten.mean, aten.relu]
# Source node to ATen node mapping:
#   out_6 => add_29
#   out_7 => mean
#   relu_12 => relu_12
# Graph fragment:
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %convolution_13), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_29, [-1, -2], True), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_24,), kwargs = {})
triton_per_fused_add_mean_relu_26 = async_compile.triton('triton_per_fused_add_mean_relu_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 8},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_relu_26(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 384)
    x1 = xindex // 384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*r2 + 3072*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.full([1, 1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (192, 3, 5, 5), (75, 25, 5, 1))
    assert_size_stride(primals_3, (192, ), (1, ))
    assert_size_stride(primals_4, (192, ), (1, ))
    assert_size_stride(primals_5, (192, ), (1, ))
    assert_size_stride(primals_6, (192, 192, 5, 5), (4800, 25, 5, 1))
    assert_size_stride(primals_7, (192, ), (1, ))
    assert_size_stride(primals_8, (192, ), (1, ))
    assert_size_stride(primals_9, (192, 192, 5, 5), (4800, 25, 5, 1))
    assert_size_stride(primals_10, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_12, (192, 192, 5, 5), (4800, 25, 5, 1))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, 192, 5, 5), (4800, 25, 5, 1))
    assert_size_stride(primals_16, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_18, (192, 192, 5, 5), (4800, 25, 5, 1))
    assert_size_stride(primals_19, (192, ), (1, ))
    assert_size_stride(primals_20, (192, ), (1, ))
    assert_size_stride(primals_21, (192, 192, 5, 5), (4800, 25, 5, 1))
    assert_size_stride(primals_22, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_23, (192, ), (1, ))
    assert_size_stride(primals_24, (192, ), (1, ))
    assert_size_stride(primals_25, (192, 192, 5, 5), (4800, 25, 5, 1))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_28, (384, 192, 5, 5), (4800, 25, 5, 1))
    assert_size_stride(primals_29, (384, ), (1, ))
    assert_size_stride(primals_30, (384, ), (1, ))
    assert_size_stride(primals_31, (384, 384, 5, 5), (9600, 25, 5, 1))
    assert_size_stride(primals_32, (384, ), (1, ))
    assert_size_stride(primals_33, (384, ), (1, ))
    assert_size_stride(primals_34, (384, 384, 5, 5), (9600, 25, 5, 1))
    assert_size_stride(primals_35, (384, ), (1, ))
    assert_size_stride(primals_36, (384, ), (1, ))
    assert_size_stride(primals_37, (384, 384, 5, 5), (9600, 25, 5, 1))
    assert_size_stride(primals_38, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (384, 384, 5, 5), (9600, 25, 5, 1))
    assert_size_stride(primals_41, (1, 384), (384, 1))
    assert_size_stride(primals_42, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((192, 3, 5, 5), (75, 1, 15, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 576, 25, grid=grid(576, 25), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((192, 192, 5, 5), (4800, 1, 960, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_6, buf2, 36864, 25, grid=grid(36864, 25), stream=stream0)
        del primals_6
        buf3 = empty_strided_cuda((192, 192, 5, 5), (4800, 1, 960, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_9, buf3, 36864, 25, grid=grid(36864, 25), stream=stream0)
        del primals_9
        buf4 = empty_strided_cuda((192, 192, 5, 5), (4800, 1, 960, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_12, buf4, 36864, 25, grid=grid(36864, 25), stream=stream0)
        del primals_12
        buf5 = empty_strided_cuda((192, 192, 5, 5), (4800, 1, 960, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_15, buf5, 36864, 25, grid=grid(36864, 25), stream=stream0)
        del primals_15
        buf6 = empty_strided_cuda((192, 192, 5, 5), (4800, 1, 960, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_18, buf6, 36864, 25, grid=grid(36864, 25), stream=stream0)
        del primals_18
        buf7 = empty_strided_cuda((192, 192, 5, 5), (4800, 1, 960, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_21, buf7, 36864, 25, grid=grid(36864, 25), stream=stream0)
        del primals_21
        buf8 = empty_strided_cuda((192, 192, 5, 5), (4800, 1, 960, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_25, buf8, 36864, 25, grid=grid(36864, 25), stream=stream0)
        del primals_25
        buf9 = empty_strided_cuda((384, 192, 5, 5), (4800, 1, 960, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_28, buf9, 73728, 25, grid=grid(73728, 25), stream=stream0)
        del primals_28
        buf10 = empty_strided_cuda((384, 384, 5, 5), (9600, 1, 1920, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_31, buf10, 147456, 25, grid=grid(147456, 25), stream=stream0)
        del primals_31
        buf11 = empty_strided_cuda((384, 384, 5, 5), (9600, 1, 1920, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_34, buf11, 147456, 25, grid=grid(147456, 25), stream=stream0)
        del primals_34
        buf12 = empty_strided_cuda((384, 384, 5, 5), (9600, 1, 1920, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_37, buf12, 147456, 25, grid=grid(147456, 25), stream=stream0)
        del primals_37
        buf13 = empty_strided_cuda((384, 384, 5, 5), (9600, 1, 1920, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_40, buf13, 147456, 25, grid=grid(147456, 25), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_5.run(buf15, primals_3, 3145728, grid=grid(3145728), stream=stream0)
        del primals_3
        buf16 = empty_strided_cuda((4, 6, 1, 1, 16, 64), (6144, 16, 24576, 24576, 1, 96), torch.float32)
        buf17 = empty_strided_cuda((4, 6, 1, 1, 16, 64), (6144, 16, 24576, 24576, 1, 96), torch.float32)
        buf18 = empty_strided_cuda((4, 6, 1, 1, 16, 64), (6144, 16, 24576, 24576, 1, 96), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_6.run(buf15, buf16, buf17, buf18, 24576, 128, grid=grid(24576), stream=stream0)
        buf19 = empty_strided_cuda((4, 6, 1, 1, 16), (96, 16, 384, 384, 1), torch.float32)
        buf20 = empty_strided_cuda((4, 6, 1, 1, 16), (96, 16, 384, 384, 1), torch.float32)
        buf21 = empty_strided_cuda((4, 6, 1, 1, 16), (96, 16, 384, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_7.run(buf16, buf17, buf18, buf19, buf20, buf21, 384, 64, grid=grid(384), stream=stream0)
        buf22 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        buf23 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        buf25 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_8.run(buf19, buf20, buf21, buf22, buf23, buf25, 24, 16, grid=grid(24), stream=stream0)
        buf26 = empty_strided_cuda((4, 192, 64, 64), (786432, 1, 12288, 192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm, relu], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_9.run(buf15, buf22, buf23, primals_4, primals_5, buf26, 3145728, grid=grid(3145728), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [dx], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, buf2, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf28 = buf18; del buf18  # reuse
        buf29 = buf17; del buf17  # reuse
        buf30 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_6.run(buf27, buf28, buf29, buf30, 24576, 128, grid=grid(24576), stream=stream0)
        buf31 = buf21; del buf21  # reuse
        buf32 = buf20; del buf20  # reuse
        buf33 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_7.run(buf28, buf29, buf30, buf31, buf32, buf33, 384, 64, grid=grid(384), stream=stream0)
        buf34 = buf23; del buf23  # reuse
        buf35 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        buf37 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_8.run(buf31, buf32, buf33, buf34, buf35, buf37, 24, 16, grid=grid(24), stream=stream0)
        buf38 = empty_strided_cuda((4, 192, 64, 64), (786432, 1, 12288, 192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_1, relu_1], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_9.run(buf27, buf34, buf35, primals_7, primals_8, buf38, 3145728, grid=grid(3145728), stream=stream0)
        del primals_8
        # Topologically Sorted Source Nodes: [dx_1], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, buf3, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf40 = buf30; del buf30  # reuse
        buf41 = buf29; del buf29  # reuse
        buf42 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_10.run(buf15, buf39, buf40, buf41, buf42, 24576, 128, grid=grid(24576), stream=stream0)
        buf43 = buf33; del buf33  # reuse
        buf44 = buf32; del buf32  # reuse
        buf45 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_7.run(buf40, buf41, buf42, buf43, buf44, buf45, 384, 64, grid=grid(384), stream=stream0)
        buf46 = buf35; del buf35  # reuse
        buf47 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        buf49 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_8.run(buf43, buf44, buf45, buf46, buf47, buf49, 24, 16, grid=grid(24), stream=stream0)
        buf50 = empty_strided_cuda((4, 192, 64, 64), (786432, 1, 12288, 192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_2, relu_2], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_11.run(buf15, buf39, buf46, buf47, primals_10, primals_11, buf50, 3145728, grid=grid(3145728), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [dx_2], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, buf4, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf52 = buf42; del buf42  # reuse
        buf53 = buf41; del buf41  # reuse
        buf54 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_6.run(buf51, buf52, buf53, buf54, 24576, 128, grid=grid(24576), stream=stream0)
        buf55 = buf45; del buf45  # reuse
        buf56 = buf44; del buf44  # reuse
        buf57 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_7.run(buf52, buf53, buf54, buf55, buf56, buf57, 384, 64, grid=grid(384), stream=stream0)
        buf58 = buf47; del buf47  # reuse
        buf59 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        buf61 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_8.run(buf55, buf56, buf57, buf58, buf59, buf61, 24, 16, grid=grid(24), stream=stream0)
        buf62 = empty_strided_cuda((4, 192, 64, 64), (786432, 1, 12288, 192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_3, relu_3], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_9.run(buf51, buf58, buf59, primals_13, primals_14, buf62, 3145728, grid=grid(3145728), stream=stream0)
        del primals_14
        # Topologically Sorted Source Nodes: [dx_3], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, buf5, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_12.run(buf64, buf15, buf39, 3145728, grid=grid(3145728), stream=stream0)
        buf65 = buf54; del buf54  # reuse
        buf66 = buf53; del buf53  # reuse
        buf67 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [group_norm_4], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_6.run(buf64, buf65, buf66, buf67, 24576, 128, grid=grid(24576), stream=stream0)
        buf68 = buf57; del buf57  # reuse
        buf69 = buf56; del buf56  # reuse
        buf70 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [group_norm_4], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_7.run(buf65, buf66, buf67, buf68, buf69, buf70, 384, 64, grid=grid(384), stream=stream0)
        buf71 = buf59; del buf59  # reuse
        buf72 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        buf74 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_4], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_8.run(buf68, buf69, buf70, buf71, buf72, buf74, 24, 16, grid=grid(24), stream=stream0)
        buf75 = empty_strided_cuda((4, 192, 64, 64), (786432, 1, 12288, 192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_4, relu_4], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_9.run(buf64, buf71, buf72, primals_16, primals_17, buf75, 3145728, grid=grid(3145728), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [dx_4], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, buf6, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf77 = buf67; del buf67  # reuse
        buf78 = buf66; del buf66  # reuse
        buf79 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_6.run(buf76, buf77, buf78, buf79, 24576, 128, grid=grid(24576), stream=stream0)
        buf80 = buf70; del buf70  # reuse
        buf81 = buf69; del buf69  # reuse
        buf82 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_7.run(buf77, buf78, buf79, buf80, buf81, buf82, 384, 64, grid=grid(384), stream=stream0)
        del buf77
        del buf78
        del buf79
        buf83 = buf72; del buf72  # reuse
        buf84 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        buf86 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_8.run(buf80, buf81, buf82, buf83, buf84, buf86, 24, 16, grid=grid(24), stream=stream0)
        del buf80
        del buf81
        del buf82
        buf87 = empty_strided_cuda((4, 192, 64, 64), (786432, 1, 12288, 192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_5, relu_5], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_9.run(buf76, buf83, buf84, primals_19, primals_20, buf87, 3145728, grid=grid(3145728), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [dx_5], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, buf7, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_13.run(buf89, buf64, 3145728, grid=grid(3145728), stream=stream0)
        buf90 = empty_strided_cuda((4, 192, 32, 32), (196608, 1, 6144, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_14.run(buf89, buf90, 786432, grid=grid(786432), stream=stream0)
        # Topologically Sorted Source Nodes: [x_s], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 384, 32, 32), (393216, 1, 12288, 384))
        buf92 = empty_strided_cuda((4, 6, 1, 1, 4, 64), (1536, 4, 6144, 6144, 1, 24), torch.float32)
        buf93 = empty_strided_cuda((4, 6, 1, 1, 4, 64), (1536, 4, 6144, 6144, 1, 24), torch.float32)
        buf94 = empty_strided_cuda((4, 6, 1, 1, 4, 64), (1536, 4, 6144, 6144, 1, 24), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf90, buf92, buf93, buf94, 6144, 128, grid=grid(6144), stream=stream0)
        buf95 = empty_strided_cuda((4, 6, 1, 1, 4), (24, 4, 96, 96, 1), torch.float32)
        buf96 = empty_strided_cuda((4, 6, 1, 1, 4), (24, 4, 96, 96, 1), torch.float32)
        buf97 = empty_strided_cuda((4, 6, 1, 1, 4), (24, 4, 96, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_16.run(buf92, buf93, buf94, buf95, buf96, buf97, 96, 64, grid=grid(96), stream=stream0)
        buf98 = buf84; del buf84  # reuse
        buf99 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        buf101 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_17.run(buf95, buf96, buf97, buf98, buf99, buf101, 24, 4, grid=grid(24), stream=stream0)
        buf102 = empty_strided_cuda((4, 192, 32, 32), (196608, 1, 6144, 192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_6, relu_6], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf90, buf98, buf99, primals_23, primals_24, buf102, 786432, grid=grid(786432), stream=stream0)
        del primals_24
        # Topologically Sorted Source Nodes: [dx_6], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, buf8, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf104 = buf94; del buf94  # reuse
        buf105 = buf93; del buf93  # reuse
        buf106 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf103, buf104, buf105, buf106, 6144, 128, grid=grid(6144), stream=stream0)
        buf107 = buf97; del buf97  # reuse
        buf108 = buf96; del buf96  # reuse
        buf109 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_16.run(buf104, buf105, buf106, buf107, buf108, buf109, 96, 64, grid=grid(96), stream=stream0)
        del buf104
        del buf105
        del buf106
        buf110 = buf99; del buf99  # reuse
        buf111 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        buf113 = empty_strided_cuda((4, 6, 1, 1), (6, 1, 24, 24), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_17.run(buf107, buf108, buf109, buf110, buf111, buf113, 24, 4, grid=grid(24), stream=stream0)
        del buf107
        del buf108
        del buf109
        buf114 = empty_strided_cuda((4, 192, 32, 32), (196608, 1, 6144, 192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_7, relu_7], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf103, buf110, buf111, primals_26, primals_27, buf114, 786432, grid=grid(786432), stream=stream0)
        del buf111
        del primals_27
        # Topologically Sorted Source Nodes: [dx_7], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, buf9, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 384, 32, 32), (393216, 1, 12288, 384))
        buf116 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_19.run(buf116, buf115, 1572864, grid=grid(1572864), stream=stream0)
        buf117 = empty_strided_cuda((4, 12, 1, 1, 4, 64), (3072, 4, 12288, 12288, 1, 48), torch.float32)
        buf118 = empty_strided_cuda((4, 12, 1, 1, 4, 64), (3072, 4, 12288, 12288, 1, 48), torch.float32)
        buf119 = empty_strided_cuda((4, 12, 1, 1, 4, 64), (3072, 4, 12288, 12288, 1, 48), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf116, buf117, buf118, buf119, 12288, 128, grid=grid(12288), stream=stream0)
        buf120 = empty_strided_cuda((4, 12, 1, 1, 4), (48, 4, 192, 192, 1), torch.float32)
        buf121 = empty_strided_cuda((4, 12, 1, 1, 4), (48, 4, 192, 192, 1), torch.float32)
        buf122 = empty_strided_cuda((4, 12, 1, 1, 4), (48, 4, 192, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf117, buf118, buf119, buf120, buf121, buf122, 192, 64, grid=grid(192), stream=stream0)
        buf123 = empty_strided_cuda((4, 12, 1, 1), (12, 1, 48, 48), torch.float32)
        buf124 = empty_strided_cuda((4, 12, 1, 1), (12, 1, 48, 48), torch.float32)
        buf126 = empty_strided_cuda((4, 12, 1, 1), (12, 1, 48, 48), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_22.run(buf120, buf121, buf122, buf123, buf124, buf126, 48, 4, grid=grid(48), stream=stream0)
        buf127 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [group_norm_8, relu_8], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_23.run(buf116, buf123, buf124, primals_29, primals_30, buf127, 1572864, grid=grid(1572864), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [dx_8], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, buf10, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 384, 32, 32), (393216, 1, 12288, 384))
        buf129 = buf119; del buf119  # reuse
        buf130 = buf118; del buf118  # reuse
        buf131 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf128, buf129, buf130, buf131, 12288, 128, grid=grid(12288), stream=stream0)
        buf132 = buf122; del buf122  # reuse
        buf133 = buf121; del buf121  # reuse
        buf134 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf129, buf130, buf131, buf132, buf133, buf134, 192, 64, grid=grid(192), stream=stream0)
        buf135 = buf124; del buf124  # reuse
        buf136 = empty_strided_cuda((4, 12, 1, 1), (12, 1, 48, 48), torch.float32)
        buf138 = empty_strided_cuda((4, 12, 1, 1), (12, 1, 48, 48), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_22.run(buf132, buf133, buf134, buf135, buf136, buf138, 48, 4, grid=grid(48), stream=stream0)
        buf139 = empty_strided_cuda((4, 384, 32, 32), (393216, 1, 12288, 384), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_9, relu_9], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_23.run(buf128, buf135, buf136, primals_32, primals_33, buf139, 1572864, grid=grid(1572864), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [dx_9], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, buf11, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 384, 32, 32), (393216, 1, 12288, 384))
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_24.run(buf141, buf116, 1572864, grid=grid(1572864), stream=stream0)
        buf142 = buf131; del buf131  # reuse
        buf143 = buf130; del buf130  # reuse
        buf144 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [group_norm_10], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf141, buf142, buf143, buf144, 12288, 128, grid=grid(12288), stream=stream0)
        buf145 = buf134; del buf134  # reuse
        buf146 = buf133; del buf133  # reuse
        buf147 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [group_norm_10], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf142, buf143, buf144, buf145, buf146, buf147, 192, 64, grid=grid(192), stream=stream0)
        buf148 = buf136; del buf136  # reuse
        buf149 = empty_strided_cuda((4, 12, 1, 1), (12, 1, 48, 48), torch.float32)
        buf151 = empty_strided_cuda((4, 12, 1, 1), (12, 1, 48, 48), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_10], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_22.run(buf145, buf146, buf147, buf148, buf149, buf151, 48, 4, grid=grid(48), stream=stream0)
        buf152 = empty_strided_cuda((4, 384, 32, 32), (393216, 1, 12288, 384), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_10, relu_10], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_23.run(buf141, buf148, buf149, primals_35, primals_36, buf152, 1572864, grid=grid(1572864), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [dx_10], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, buf12, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 384, 32, 32), (393216, 1, 12288, 384))
        buf154 = buf144; del buf144  # reuse
        buf155 = buf143; del buf143  # reuse
        buf156 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf153, buf154, buf155, buf156, 12288, 128, grid=grid(12288), stream=stream0)
        buf157 = buf147; del buf147  # reuse
        buf158 = buf146; del buf146  # reuse
        buf159 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf154, buf155, buf156, buf157, buf158, buf159, 192, 64, grid=grid(192), stream=stream0)
        del buf154
        del buf155
        buf160 = buf149; del buf149  # reuse
        buf161 = empty_strided_cuda((4, 12, 1, 1), (12, 1, 48, 48), torch.float32)
        buf163 = empty_strided_cuda((4, 12, 1, 1), (12, 1, 48, 48), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_22.run(buf157, buf158, buf159, buf160, buf161, buf163, 48, 4, grid=grid(48), stream=stream0)
        del buf157
        del buf158
        del buf159
        buf164 = empty_strided_cuda((4, 384, 32, 32), (393216, 1, 12288, 384), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_11, relu_11], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_23.run(buf153, buf160, buf161, primals_38, primals_39, buf164, 1572864, grid=grid(1572864), stream=stream0)
        del buf161
        del primals_39
        # Topologically Sorted Source Nodes: [dx_11], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, buf13, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 384, 32, 32), (393216, 1, 12288, 384))
        buf166 = reinterpret_tensor(buf156, (4, 384, 1, 1, 8), (3072, 1, 12288, 12288, 384), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [out_6, out_7], Original ATen: [aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mean_25.run(buf141, buf165, buf166, 12288, 128, grid=grid(12288), stream=stream0)
        del buf165
        buf167 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 1536, 1536), torch.float32)
        buf168 = reinterpret_tensor(buf167, (4, 384), (384, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [out_6, out_7, relu_12], Original ATen: [aten.add, aten.mean, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_relu_26.run(buf168, buf166, 1536, 8, grid=grid(1536), stream=stream0)
        del buf166
        buf170 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_42, buf168, reinterpret_tensor(primals_41, (384, 1), (1, 384), 0), alpha=1, beta=1, out=buf170)
        del primals_42
    return (buf170, buf0, buf1, primals_4, buf2, primals_7, buf3, primals_10, buf4, primals_13, buf5, primals_16, buf6, primals_19, buf7, primals_22, primals_23, buf8, primals_26, buf9, primals_29, buf10, primals_32, buf11, primals_35, buf12, primals_38, buf13, buf15, reinterpret_tensor(buf22, (4, 6), (6, 1), 0), reinterpret_tensor(buf25, (4, 6), (6, 1), 0), buf26, buf27, reinterpret_tensor(buf34, (4, 6), (6, 1), 0), reinterpret_tensor(buf37, (4, 6), (6, 1), 0), buf38, buf39, reinterpret_tensor(buf46, (4, 6), (6, 1), 0), reinterpret_tensor(buf49, (4, 6), (6, 1), 0), buf50, buf51, reinterpret_tensor(buf58, (4, 6), (6, 1), 0), reinterpret_tensor(buf61, (4, 6), (6, 1), 0), buf62, buf64, reinterpret_tensor(buf71, (4, 6), (6, 1), 0), reinterpret_tensor(buf74, (4, 6), (6, 1), 0), buf75, buf76, reinterpret_tensor(buf83, (4, 6), (6, 1), 0), reinterpret_tensor(buf86, (4, 6), (6, 1), 0), buf87, buf89, buf90, reinterpret_tensor(buf98, (4, 6), (6, 1), 0), reinterpret_tensor(buf101, (4, 6), (6, 1), 0), buf102, buf103, reinterpret_tensor(buf110, (4, 6), (6, 1), 0), reinterpret_tensor(buf113, (4, 6), (6, 1), 0), buf114, buf116, reinterpret_tensor(buf123, (4, 12), (12, 1), 0), reinterpret_tensor(buf126, (4, 12), (12, 1), 0), buf127, buf128, reinterpret_tensor(buf135, (4, 12), (12, 1), 0), reinterpret_tensor(buf138, (4, 12), (12, 1), 0), buf139, buf141, reinterpret_tensor(buf148, (4, 12), (12, 1), 0), reinterpret_tensor(buf151, (4, 12), (12, 1), 0), buf152, buf153, reinterpret_tensor(buf160, (4, 12), (12, 1), 0), reinterpret_tensor(buf163, (4, 12), (12, 1), 0), buf164, buf168, primals_41, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((192, 3, 5, 5), (75, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((192, 192, 5, 5), (4800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, 192, 5, 5), (4800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((192, 192, 5, 5), (4800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, 192, 5, 5), (4800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((192, 192, 5, 5), (4800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((192, 192, 5, 5), (4800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, 192, 5, 5), (4800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, 192, 5, 5), (4800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((384, 384, 5, 5), (9600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, 384, 5, 5), (9600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((384, 384, 5, 5), (9600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, 384, 5, 5), (9600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
