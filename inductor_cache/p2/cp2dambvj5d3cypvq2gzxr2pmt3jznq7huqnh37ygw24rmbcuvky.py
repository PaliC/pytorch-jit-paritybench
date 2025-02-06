# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/z5/cz5lcpzem5jshoxsyxcxgxhiha6q7ktzqnt2hpqcbwodal3x5j3j.py
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
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
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


# kernel path: inductor_cache/l6/cl6giekbizdbnf3zd5gwfgrns5u4drugr2lvufwr3ipnzsmtpimy.py
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
    ynumel = 32768
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


# kernel path: inductor_cache/n4/cn4qbmztxrtdjckkhya25vms3xu5rvrlec42b5y5ojopl6ngwgh7.py
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
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/4g/c4gzvkyepx3rappomuisjehfkptoui4ssyogeeec6yowrbw7kavn.py
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


# kernel path: inductor_cache/o4/co4od6po6wgrmh6qlxlc6xamyctm62zvpgkhp4zkhybflikht4u2.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => gt, mul_3, where
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/ek/cekv4lhpzv6wjvsdoyy5ki6crdu7yf3nwppbqakpxhnyopwdypxp.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_5 => add_3, mul_5, mul_6, sub_1
#   input_6 => gt_1, mul_7, where_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.1), kwargs = {})
#   %where_1 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_3, %mul_7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci4r6vk7igfdhkeveqzw7z7fqz7dcca4o67f33tfe3gqwczvjs2t.py
# Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_8 => add_5, mul_10, mul_9, sub_2
#   input_9 => gt_2, mul_11, where_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_5, 0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.1), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_5, %mul_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/q7/cq7wge5z5oqhdnws2bqcykzpth2pf2tauikosgh72fhuvnje2g73.py
# Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_11 => add_7, mul_13, mul_14, sub_3
#   input_12 => gt_3, mul_15, where_3
#   input_13 => add_8
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 0), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.1), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_7, %mul_15), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_3, %where_1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x1 + 64*y0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1 + 64*y0), xmask, eviction_policy='evict_last')
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr1 + (y2 + 1024*x1 + 65536*y3), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gh/cghnnxxedcddv43ipx4hld7btbk4d76e6uqka5td42fdcmlm4xwp.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_14 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %primals_22, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_11 = async_compile.triton('triton_poi_fused_convolution_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 65536*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/uj/cujhdus4nree7u7p4ijgvfakvscmipppysqulwakmbtd6dr4ntzx.py
# Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_15 => add_10, mul_17, mul_18, sub_4
#   input_16 => gt_4, mul_19, where_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_37), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_10, 0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.1), kwargs = {})
#   %where_4 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %add_10, %mul_19), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/ec/cecillnbr55wzhl2tdyqreefljk6lfpvyrhm6mz22vxnbiju5kp3.py
# Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_18 => add_12, mul_21, mul_22, sub_5
#   input_19 => gt_5, mul_23, where_5
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_47), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_12, 0), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, 0.1), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_12, %mul_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/px/cpxxxzhr7t3l7jnatmwagi4tzd6knt4w64oaqgedvxl2jjnii23n.py
# Topologically Sorted Source Nodes: [input_21, input_22, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_21 => add_14, mul_25, mul_26, sub_6
#   input_22 => gt_6, mul_27, where_6
#   input_23 => add_15
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_53), kwargs = {})
#   %add_14 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_55), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_14, 0), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, 0.1), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_14, %mul_27), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_6, %where_4), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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
    tmp21 = tl.load(in_ptr5 + (x2), None)
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/tx/ctx5it4zwm4i5kce7vrj22bnocsvfsfdgxw3jnbdd4rma2j3yynu.py
# Topologically Sorted Source Nodes: [input_28, input_29, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_28 => add_19, mul_33, mul_34, sub_8
#   input_29 => gt_8, mul_35, where_8
#   input_30 => add_20
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %unsqueeze_69), kwargs = {})
#   %add_19 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %unsqueeze_71), kwargs = {})
#   %gt_8 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_19, 0), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, 0.1), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %add_19, %mul_35), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_8, %add_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x1 + 128*y0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1 + 128*y0), xmask, eviction_policy='evict_last')
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr1 + (y2 + 256*x1 + 32768*y3), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qa/cqalrzp3cafqhp3urxvtnedjh4vbv2apj7tgezok5xt3w4gatmw4.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_31 => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_20, %primals_47, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_16 = async_compile.triton('triton_poi_fused_convolution_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 32768*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/gw/cgwb2tawpstbkfpd5lwozicd75get76uttsueyxlbvilyl4opjok.py
# Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_32 => add_22, mul_37, mul_38, sub_9
#   input_33 => gt_9, mul_39, where_9
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_77), kwargs = {})
#   %add_22 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_79), kwargs = {})
#   %gt_9 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_22, 0), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.1), kwargs = {})
#   %where_9 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %add_22, %mul_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/zm/czmsifxf6k5sziehasmriqx6wjpqckzobxkeleik3aecj54wsopj.py
# Topologically Sorted Source Nodes: [input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_35 => add_24, mul_41, mul_42, sub_10
#   input_36 => gt_10, mul_43, where_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %unsqueeze_85), kwargs = {})
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %unsqueeze_87), kwargs = {})
#   %gt_10 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_24, 0), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, 0.1), kwargs = {})
#   %where_10 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_10, %add_24, %mul_43), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/uj/cujwlqac4qws4pipn5vvzndv6ha4klmlbj6urrhyxiugpyiqm3p4.py
# Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_38 => add_26, mul_45, mul_46, sub_11
#   input_39 => gt_11, mul_47, where_11
#   input_40 => add_27
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %unsqueeze_93), kwargs = {})
#   %add_26 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %unsqueeze_95), kwargs = {})
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_26, 0), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, 0.1), kwargs = {})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_11, %add_26, %mul_47), kwargs = {})
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_11, %where_9), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    tmp21 = tl.load(in_ptr5 + (x2), None)
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/r7/cr7g5ygfs4tf2s6uj5s4tzxadhlc45i2d3qyvr6rd6i62ekrylhz.py
# Topologically Sorted Source Nodes: [input_87, input_88, input_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_87 => add_61, mul_101, mul_102, sub_25
#   input_88 => gt_25, mul_103, where_25
#   input_89 => add_62
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_101, %unsqueeze_205), kwargs = {})
#   %add_61 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_102, %unsqueeze_207), kwargs = {})
#   %gt_25 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_61, 0), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_61, 0.1), kwargs = {})
#   %where_25 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_25, %add_61, %mul_103), kwargs = {})
#   %add_62 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_25, %add_57), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = (yindex % 64)
    y3 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1 + 256*y0), xmask & ymask)
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr1 + (y2 + 64*x1 + 16384*y3), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/hq/chqadch6q3ge2jzrwuan266jpqzztsrj4s6zbhpurbbzdo7hiv7m.py
# Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_90 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_62, %primals_132, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_21 = async_compile.triton('triton_poi_fused_convolution_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 16384*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2c/c2cabqbeogpkhxhblkknycpw7vcxkb5yt245nvo7l2damsuf7tcr.py
# Topologically Sorted Source Nodes: [input_91, input_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_91 => add_64, mul_105, mul_106, sub_26
#   input_92 => gt_26, mul_107, where_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_105, %unsqueeze_213), kwargs = {})
#   %add_64 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %unsqueeze_215), kwargs = {})
#   %gt_26 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_64, 0), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_64, 0.1), kwargs = {})
#   %where_26 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_26, %add_64, %mul_107), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/ft/cft5endd5edqy4njpx3dh6gyycuovnmtymddaurcgcya4yqrnjyy.py
# Topologically Sorted Source Nodes: [input_94, input_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_94 => add_66, mul_109, mul_110, sub_27
#   input_95 => gt_27, mul_111, where_27
# Graph fragment:
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_221), kwargs = {})
#   %add_66 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_223), kwargs = {})
#   %gt_27 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_66, 0), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_66, 0.1), kwargs = {})
#   %where_27 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_27, %add_66, %mul_111), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/my/cmygxpmlix7ignkk52kfwuxusqrsftsvxwrep7ghtzblamathcht.py
# Topologically Sorted Source Nodes: [input_97, input_98, input_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_97 => add_68, mul_113, mul_114, sub_28
#   input_98 => gt_28, mul_115, where_28
#   input_99 => add_69
# Graph fragment:
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %unsqueeze_229), kwargs = {})
#   %add_68 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %unsqueeze_231), kwargs = {})
#   %gt_28 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_68, 0), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_68, 0.1), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_28, %add_68, %mul_115), kwargs = {})
#   %add_69 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_28, %where_26), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x2), None)
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/ke/ckewedlycibszarnear7noymo4onxfxh45syg4qtydg3cpflk25w.py
# Topologically Sorted Source Nodes: [input_146, input_147, input_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_146 => add_103, mul_169, mul_170, sub_42
#   input_147 => gt_42, mul_171, where_42
#   input_148 => add_104
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_169, %unsqueeze_341), kwargs = {})
#   %add_103 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_170, %unsqueeze_343), kwargs = {})
#   %gt_42 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_103, 0), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_103, 0.1), kwargs = {})
#   %where_42 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_42, %add_103, %mul_171), kwargs = {})
#   %add_104 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_42, %add_99), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = (yindex % 16)
    y3 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr1 + (y2 + 16*x1 + 8192*y3), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/6b/c6brjuci2sf4iwity54d5cs52omauniwu4g3dd6xojauavyhiagj.py
# Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_149 => convolution_43
# Graph fragment:
#   %convolution_43 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_104, %primals_217, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_26 = async_compile.triton('triton_poi_fused_convolution_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 8192*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6r/c6rk7w333kw6i4r4eqiokevklx3cf5zrebxbzkpnul3tttxu6ocn.py
# Topologically Sorted Source Nodes: [input_150, input_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_150 => add_106, mul_173, mul_174, sub_43
#   input_151 => gt_43, mul_175, where_43
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_173, %unsqueeze_349), kwargs = {})
#   %add_106 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_174, %unsqueeze_351), kwargs = {})
#   %gt_43 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_106, 0), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_106, 0.1), kwargs = {})
#   %where_43 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_43, %add_106, %mul_175), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/xj/cxjwqfy6vcagktv7cowtokcmknytmgmt5xeramyfkqmsrwjlligu.py
# Topologically Sorted Source Nodes: [input_153, input_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_153 => add_108, mul_177, mul_178, sub_44
#   input_154 => gt_44, mul_179, where_44
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_177 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_177, %unsqueeze_357), kwargs = {})
#   %add_108 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_178, %unsqueeze_359), kwargs = {})
#   %gt_44 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_108, 0), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_108, 0.1), kwargs = {})
#   %where_44 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_44, %add_108, %mul_179), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/an/cani5dvuut7fthgqby4tk3ao7hd4d2c3zeazczs2lytslnvu6pyq.py
# Topologically Sorted Source Nodes: [input_156, input_157, input_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_156 => add_110, mul_181, mul_182, sub_45
#   input_157 => gt_45, mul_183, where_45
#   input_158 => add_111
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_181, %unsqueeze_365), kwargs = {})
#   %add_110 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %unsqueeze_367), kwargs = {})
#   %gt_45 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_110, 0), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, 0.1), kwargs = {})
#   %where_45 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_45, %add_110, %mul_183), kwargs = {})
#   %add_111 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_45, %where_43), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tmp21 = tl.load(in_ptr5 + (x2), None)
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/jb/cjbrjo77mymvbhl54s7auw2bnaydty4tuqihkvchqefzzue6oor3.py
# Topologically Sorted Source Nodes: [input_177, input_178, input_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_177 => add_125, mul_205, mul_206, sub_51
#   input_178 => gt_51, mul_207, where_51
#   input_179 => add_126
# Graph fragment:
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_409), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, %unsqueeze_413), kwargs = {})
#   %add_125 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_206, %unsqueeze_415), kwargs = {})
#   %gt_51 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_125, 0), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_125, 0.1), kwargs = {})
#   %where_51 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_51, %add_125, %mul_207), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_51, %add_121), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = (yindex % 4)
    y3 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x1 + 1024*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1 + 1024*y0), xmask & ymask, eviction_policy='evict_last')
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
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr1 + (y2 + 4*x1 + 4096*y3), tmp22, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, ), (1, ))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (256, ), (1, ))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (128, ), (1, ))
    assert_size_stride(primals_75, (128, ), (1, ))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_82, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (256, ), (1, ))
    assert_size_stride(primals_121, (256, ), (1, ))
    assert_size_stride(primals_122, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_133, (512, ), (1, ))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (512, ), (1, ))
    assert_size_stride(primals_136, (512, ), (1, ))
    assert_size_stride(primals_137, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_138, (256, ), (1, ))
    assert_size_stride(primals_139, (256, ), (1, ))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (256, ), (1, ))
    assert_size_stride(primals_142, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, ), (1, ))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_148, (256, ), (1, ))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_151, (256, ), (1, ))
    assert_size_stride(primals_152, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_153, (512, ), (1, ))
    assert_size_stride(primals_154, (512, ), (1, ))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (512, ), (1, ))
    assert_size_stride(primals_157, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_161, (256, ), (1, ))
    assert_size_stride(primals_162, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_163, (512, ), (1, ))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_168, (256, ), (1, ))
    assert_size_stride(primals_169, (256, ), (1, ))
    assert_size_stride(primals_170, (256, ), (1, ))
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_174, (512, ), (1, ))
    assert_size_stride(primals_175, (512, ), (1, ))
    assert_size_stride(primals_176, (512, ), (1, ))
    assert_size_stride(primals_177, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_178, (256, ), (1, ))
    assert_size_stride(primals_179, (256, ), (1, ))
    assert_size_stride(primals_180, (256, ), (1, ))
    assert_size_stride(primals_181, (256, ), (1, ))
    assert_size_stride(primals_182, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_183, (512, ), (1, ))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (512, ), (1, ))
    assert_size_stride(primals_186, (512, ), (1, ))
    assert_size_stride(primals_187, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_188, (256, ), (1, ))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_190, (256, ), (1, ))
    assert_size_stride(primals_191, (256, ), (1, ))
    assert_size_stride(primals_192, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_193, (512, ), (1, ))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, ), (1, ))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (256, ), (1, ))
    assert_size_stride(primals_202, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_207, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_209, (256, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_212, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_216, (512, ), (1, ))
    assert_size_stride(primals_217, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_219, (1024, ), (1, ))
    assert_size_stride(primals_220, (1024, ), (1, ))
    assert_size_stride(primals_221, (1024, ), (1, ))
    assert_size_stride(primals_222, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (512, ), (1, ))
    assert_size_stride(primals_225, (512, ), (1, ))
    assert_size_stride(primals_226, (512, ), (1, ))
    assert_size_stride(primals_227, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (1024, ), (1, ))
    assert_size_stride(primals_230, (1024, ), (1, ))
    assert_size_stride(primals_231, (1024, ), (1, ))
    assert_size_stride(primals_232, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_233, (512, ), (1, ))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (512, ), (1, ))
    assert_size_stride(primals_236, (512, ), (1, ))
    assert_size_stride(primals_237, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_239, (1024, ), (1, ))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_242, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_243, (512, ), (1, ))
    assert_size_stride(primals_244, (512, ), (1, ))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_248, (1024, ), (1, ))
    assert_size_stride(primals_249, (1024, ), (1, ))
    assert_size_stride(primals_250, (1024, ), (1, ))
    assert_size_stride(primals_251, (1024, ), (1, ))
    assert_size_stride(primals_252, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_253, (512, ), (1, ))
    assert_size_stride(primals_254, (512, ), (1, ))
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_259, (1024, ), (1, ))
    assert_size_stride(primals_260, (1024, ), (1, ))
    assert_size_stride(primals_261, (1024, ), (1, ))
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
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_7, buf2, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_7
        buf3 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_17, buf3, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_17
        buf4 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_22, buf4, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_22
        buf5 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_32, buf5, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_32
        buf6 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_42, buf6, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_42
        buf7 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_47, buf7, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_47
        buf8 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_57, buf8, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_57
        buf9 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_67, buf9, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_67
        buf10 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_77, buf10, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_77
        buf11 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_87, buf11, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_87
        buf12 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_97, buf12, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_97
        buf13 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_107, buf13, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_107
        buf14 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_117, buf14, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_117
        buf15 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_127, buf15, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_127
        buf16 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_132, buf16, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_132
        buf17 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_142, buf17, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_142
        buf18 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_152, buf18, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_152
        buf19 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_162, buf19, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_162
        buf20 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_172, buf20, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_172
        buf21 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_182, buf21, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_182
        buf22 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_192, buf22, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_192
        buf23 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_202, buf23, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_202
        buf24 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_212, buf24, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_212
        buf25 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_217, buf25, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_217
        buf26 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_227, buf26, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_227
        buf27 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_237, buf27, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_237
        buf28 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_247, buf28, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_247
        buf29 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_257, buf29, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_257
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf1, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 32, 64, 64), (131072, 1, 2048, 32))
        buf31 = empty_strided_cuda((4, 32, 64, 64), (131072, 1, 2048, 32), torch.float32)
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7.run(buf32, buf30, primals_3, primals_4, primals_5, primals_6, 524288, grid=grid(524288), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf34 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_8.run(buf35, buf33, primals_8, primals_9, primals_10, primals_11, 262144, grid=grid(262144), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf37 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_9.run(buf38, buf36, primals_13, primals_14, primals_15, primals_16, 131072, grid=grid(131072), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf41 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_10.run(buf39, primals_18, primals_19, primals_20, primals_21, buf35, buf41, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf42 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_11.run(buf41, buf42, 256, 1024, grid=grid(256, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 128, 16, 16), (32768, 1, 2048, 128))
        del buf42
        buf44 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12.run(buf45, buf43, primals_23, primals_24, primals_25, primals_26, 131072, grid=grid(131072), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf47 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13.run(buf48, buf46, primals_28, primals_29, primals_30, primals_31, 65536, grid=grid(65536), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf50 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_21, input_22, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_14.run(buf51, buf49, primals_33, primals_34, primals_35, primals_36, buf45, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf53 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf54 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13.run(buf54, buf52, primals_38, primals_39, primals_40, primals_41, 65536, grid=grid(65536), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf57 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, input_29, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_15.run(buf55, primals_43, primals_44, primals_45, primals_46, buf51, buf57, 1024, 128, grid=grid(1024, 128), stream=stream0)
        buf58 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_16.run(buf57, buf58, 512, 256, grid=grid(512, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 256, 8, 8), (16384, 1, 2048, 256))
        del buf58
        buf60 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17.run(buf61, buf59, primals_48, primals_49, primals_50, primals_51, 65536, grid=grid(65536), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf63 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf64, buf62, primals_53, primals_54, primals_55, primals_56, 32768, grid=grid(32768), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf66 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19.run(buf67, buf65, primals_58, primals_59, primals_60, primals_61, buf61, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf69 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [input_42, input_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf70, buf68, primals_63, primals_64, primals_65, primals_66, 32768, grid=grid(32768), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf72 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_45, input_46, input_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19.run(buf73, buf71, primals_68, primals_69, primals_70, primals_71, buf67, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf75 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [input_49, input_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf76, buf74, primals_73, primals_74, primals_75, primals_76, 32768, grid=grid(32768), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf78 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19.run(buf79, buf77, primals_78, primals_79, primals_80, primals_81, buf73, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf81 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf82 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [input_56, input_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf82, buf80, primals_83, primals_84, primals_85, primals_86, 32768, grid=grid(32768), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf84 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_59, input_60, input_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19.run(buf85, buf83, primals_88, primals_89, primals_90, primals_91, buf79, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf87 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf88 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [input_63, input_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf88, buf86, primals_93, primals_94, primals_95, primals_96, 32768, grid=grid(32768), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf90 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [input_66, input_67, input_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19.run(buf91, buf89, primals_98, primals_99, primals_100, primals_101, buf85, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf93 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [input_70, input_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf94, buf92, primals_103, primals_104, primals_105, primals_106, 32768, grid=grid(32768), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf96 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [input_73, input_74, input_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19.run(buf97, buf95, primals_108, primals_109, primals_110, primals_111, buf91, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf99 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [input_77, input_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf100, buf98, primals_113, primals_114, primals_115, primals_116, 32768, grid=grid(32768), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf102 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [input_80, input_81, input_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_19.run(buf103, buf101, primals_118, primals_119, primals_120, primals_121, buf97, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf105 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [input_84, input_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf106, buf104, primals_123, primals_124, primals_125, primals_126, 32768, grid=grid(32768), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf109 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_87, input_88, input_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_20.run(buf107, primals_128, primals_129, primals_130, primals_131, buf103, buf109, 256, 256, grid=grid(256, 256), stream=stream0)
        buf110 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_21.run(buf109, buf110, 1024, 64, grid=grid(1024, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, buf16, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf110
        buf112 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [input_91, input_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22.run(buf113, buf111, primals_133, primals_134, primals_135, primals_136, 32768, grid=grid(32768), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf115 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf116 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [input_94, input_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf116, buf114, primals_138, primals_139, primals_140, primals_141, 16384, grid=grid(16384), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf118 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [input_97, input_98, input_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24.run(buf119, buf117, primals_143, primals_144, primals_145, primals_146, buf113, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_100], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf121 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [input_101, input_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf122, buf120, primals_148, primals_149, primals_150, primals_151, 16384, grid=grid(16384), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf124 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [input_104, input_105, input_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24.run(buf125, buf123, primals_153, primals_154, primals_155, primals_156, buf119, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_107], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf127 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [input_108, input_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf128, buf126, primals_158, primals_159, primals_160, primals_161, 16384, grid=grid(16384), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf130 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [input_111, input_112, input_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24.run(buf131, buf129, primals_163, primals_164, primals_165, primals_166, buf125, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf133 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf134 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [input_115, input_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf134, buf132, primals_168, primals_169, primals_170, primals_171, 16384, grid=grid(16384), stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf136 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [input_118, input_119, input_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24.run(buf137, buf135, primals_173, primals_174, primals_175, primals_176, buf131, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_121], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf139 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf140 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [input_122, input_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf140, buf138, primals_178, primals_179, primals_180, primals_181, 16384, grid=grid(16384), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf142 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf143 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [input_125, input_126, input_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24.run(buf143, buf141, primals_183, primals_184, primals_185, primals_186, buf137, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_128], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf145 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf146 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [input_129, input_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf146, buf144, primals_188, primals_189, primals_190, primals_191, 16384, grid=grid(16384), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf148 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf149 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [input_132, input_133, input_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24.run(buf149, buf147, primals_193, primals_194, primals_195, primals_196, buf143, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf151 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf152 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [input_136, input_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf152, buf150, primals_198, primals_199, primals_200, primals_201, 16384, grid=grid(16384), stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [input_138], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf154 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [input_139, input_140, input_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_24.run(buf155, buf153, primals_203, primals_204, primals_205, primals_206, buf149, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_142], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf157 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [input_143, input_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf158, buf156, primals_208, primals_209, primals_210, primals_211, 16384, grid=grid(16384), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf161 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_146, input_147, input_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_25.run(buf159, primals_213, primals_214, primals_215, primals_216, buf155, buf161, 64, 512, grid=grid(64, 512), stream=stream0)
        buf162 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_26.run(buf161, buf162, 2048, 16, grid=grid(2048, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, buf25, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        del buf162
        buf164 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf165 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [input_150, input_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_27.run(buf165, buf163, primals_218, primals_219, primals_220, primals_221, 16384, grid=grid(16384), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [input_152], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf167 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf168 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [input_153, input_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_28.run(buf168, buf166, primals_223, primals_224, primals_225, primals_226, 8192, grid=grid(8192), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [input_155], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf170 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf171 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [input_156, input_157, input_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_29.run(buf171, buf169, primals_228, primals_229, primals_230, primals_231, buf165, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf173 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf174 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [input_160, input_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_28.run(buf174, buf172, primals_233, primals_234, primals_235, primals_236, 8192, grid=grid(8192), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf176 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf177 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [input_163, input_164, input_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_29.run(buf177, buf175, primals_238, primals_239, primals_240, primals_241, buf171, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf179 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_167, input_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_28.run(buf180, buf178, primals_243, primals_244, primals_245, primals_246, 8192, grid=grid(8192), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [input_169], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf182 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf183 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [input_170, input_171, input_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_29.run(buf183, buf181, primals_248, primals_249, primals_250, primals_251, buf177, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_173], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf185 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf186 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [input_174, input_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_28.run(buf186, buf184, primals_253, primals_254, primals_255, primals_256, 8192, grid=grid(8192), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [input_176], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf189 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_177, input_178, input_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_30.run(buf187, primals_258, primals_259, primals_260, primals_261, buf183, buf189, 16, 1024, grid=grid(16, 1024), stream=stream0)
    return (buf41, buf57, buf109, buf161, buf189, buf0, buf1, primals_3, primals_4, primals_5, buf2, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, buf3, primals_18, primals_19, primals_20, primals_21, buf4, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, buf5, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, buf6, primals_43, primals_44, primals_45, primals_46, buf7, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, buf8, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, buf9, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, buf10, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, buf11, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, buf12, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, buf13, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, buf14, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, buf15, primals_128, primals_129, primals_130, primals_131, buf16, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, buf17, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, buf18, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, buf19, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, buf20, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, buf21, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, buf22, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, buf23, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, buf24, primals_213, primals_214, primals_215, primals_216, buf25, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, buf26, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, buf27, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, buf28, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, buf29, primals_258, primals_259, primals_260, primals_261, buf30, buf32, buf33, buf35, buf36, buf38, buf39, buf41, buf43, buf45, buf46, buf48, buf49, buf51, buf52, buf54, buf55, buf57, buf59, buf61, buf62, buf64, buf65, buf67, buf68, buf70, buf71, buf73, buf74, buf76, buf77, buf79, buf80, buf82, buf83, buf85, buf86, buf88, buf89, buf91, buf92, buf94, buf95, buf97, buf98, buf100, buf101, buf103, buf104, buf106, buf107, buf109, buf111, buf113, buf114, buf116, buf117, buf119, buf120, buf122, buf123, buf125, buf126, buf128, buf129, buf131, buf132, buf134, buf135, buf137, buf138, buf140, buf141, buf143, buf144, buf146, buf147, buf149, buf150, buf152, buf153, buf155, buf156, buf158, buf159, buf161, buf163, buf165, buf166, buf168, buf169, buf171, buf172, buf174, buf175, buf177, buf178, buf180, buf181, buf183, buf184, buf186, buf187, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
