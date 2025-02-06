# AOT ID: ['86_forward']
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


# kernel path: inductor_cache/lc/clcxu7vsukg7yhskixzvn5mxcvkd2w55ns3ha3u3tqmqqfqng5mi.py
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
    size_hints={'y': 128, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 4*x2 + 196*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/kx/ckxwzs42unzjm4hbkrn2nggk65sxlou7klkj6ooaow3hhxc2bpqx.py
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
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/gd/cgdw6q73jcandmabgcgiyoxvo5x7ujex77af7jsltqagydbl4alk.py
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
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/n5/cn5aahiul4w6pzk7n4q4lmu6yb65rd34adiaysgggurgl2ltdi5z.py
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
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/3y/c3y6g73uj6w62ahjw35ny3qtiepr3f75boorkraxg6xspeobiue3.py
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


# kernel path: inductor_cache/cz/ccz3syobdwf2qjusl7u5yxorsohclzq4rey7ak7wndcwtjxivziu.py
# Topologically Sorted Source Nodes: [input_1, mask, mul], Original ATen: [aten.reflection_pad2d, aten.mul]
# Source node to ATen node mapping:
#   input_1 => _unsafe_index, _unsafe_index_1
#   mask => _unsafe_index_2, _unsafe_index_3
#   mul => mul
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_2, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %sub_1]), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_1, %_unsafe_index_3), kwargs = {})
triton_poi_fused_mul_reflection_pad2d_5 = async_compile.triton('triton_poi_fused_mul_reflection_pad2d_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_reflection_pad2d_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_reflection_pad2d_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 100
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 10)
    x3 = xindex // 10
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-3) + x2))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-3) + x3))) + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-3) + x2))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-3) + x3))) + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (y0 + 4*x5 + 400*y1), tmp2, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 4*x5 + 400*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/y7/cy7eh47jh723mutrb6adu7umsubkrc4t4ic2isrpbq6oko2p3l6k.py
# Topologically Sorted Source Nodes: [output, no_update_holes, mask_sum, sub, truediv, output_pre, output_1, new_mask, new_mask_1, input_2, input_3, mul_1], Original ATen: [aten.convolution, aten.eq, aten.masked_fill, aten.sub, aten.div, aten.add, aten.ones_like, aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
# Source node to ATen node mapping:
#   input_2 => add_2, mul_2, mul_3, sub_9
#   input_3 => relu
#   mask_sum => full_default, where
#   mul_1 => mul_4
#   new_mask => full_default_2
#   new_mask_1 => where_2
#   no_update_holes => eq
#   output => convolution
#   output_1 => full_default_1, where_1
#   output_pre => add
#   sub => sub_8
#   truediv => div
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul, %primals_3, %primals_4, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %eq : [num_users=3] = call_function[target=torch.ops.aten.eq.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %convolution_1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %expand), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_8, %where), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %expand), kwargs = {})
#   %full_default_1 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_1, %add), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 32, 4, 4], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_1, %full_default_2), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %unsqueeze_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_3), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %unsqueeze_5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_2,), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %where_2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 0.0
    tmp5 = tmp3 == tmp4
    tmp6 = tmp2 - tmp1
    tmp7 = 1.0
    tmp8 = tl.where(tmp5, tmp7, tmp3)
    tmp9 = tmp6 / tmp8
    tmp10 = tmp9 + tmp1
    tmp11 = tl.where(tmp5, tmp4, tmp10)
    tmp13 = tmp11 - tmp12
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp18 / tmp17
    tmp20 = tmp19 * tmp7
    tmp21 = tmp13 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tmp28 = tl.where(tmp5, tmp4, tmp7)
    tmp29 = tmp27 * tmp28
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp29, xmask)
    tl.store(out_ptr1 + (x2), tmp28, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lz/clzqx6qb57jaodxn37o3a462sbseelfoculpeselqjqwc6nfazgu.py
# Topologically Sorted Source Nodes: [mask_sum, output_1, output_2, no_update_holes_1, mask_sum_1, sub_1, truediv_1, output_pre_1, output_3, new_mask_2, new_mask_3, input_4, input_5, mul_2], Original ATen: [aten.masked_fill, aten.convolution, aten.eq, aten.sub, aten.div, aten.add, aten.ones_like, aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
# Source node to ATen node mapping:
#   input_4 => add_5, mul_6, mul_7, sub_11
#   input_5 => relu_1
#   mask_sum => full_default
#   mask_sum_1 => where_3
#   mul_2 => mul_8
#   new_mask_2 => full_default_6
#   new_mask_3 => where_5
#   no_update_holes_1 => eq_1
#   output_1 => full_default_1
#   output_2 => convolution_2
#   output_3 => where_4
#   output_pre_1 => add_3
#   sub_1 => sub_10
#   truediv_1 => div_1
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_4, %primals_10, %primals_11, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %eq_1 : [num_users=4] = call_function[target=torch.ops.aten.eq.Scalar](args = (%convolution_3, 0), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default, %convolution_3), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %expand_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_10, %where_3), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, %expand_1), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_1, %add_3), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 64, 2, 2], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_1, %full_default_6), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_4, %unsqueeze_9), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_11), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_13), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_1, %where_5), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 - tmp6
    tmp9 = tmp8 / tmp4
    tmp10 = tmp9 + tmp6
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tl.where(tmp2, tmp1, tmp3)
    tmp14 = tmp11 - tmp13
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tl.full([1], 1, tl.int32)
    tmp20 = tmp19 / tmp18
    tmp21 = tmp20 * tmp3
    tmp22 = tmp14 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full([1], 0, tl.int32)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tmp29 = tmp28 * tmp12
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
    tl.store(in_out_ptr1 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
    tl.store(out_ptr2 + (x0), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ha/charsq46tmsqmlhqrrvztyxv7oybhlnrymyydvzxrpmrrskyarv4.py
# Topologically Sorted Source Nodes: [mask_sum, output_1, output_4, no_update_holes_2, mask_sum_2, sub_2, truediv_2, output_pre_2, output_5, new_mask_4, new_mask_5, input_6, input_7, mul_3], Original ATen: [aten.masked_fill, aten.convolution, aten.eq, aten.sub, aten.div, aten.add, aten.ones_like, aten._native_batch_norm_legit_no_training, aten.relu, aten.mul, aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_6 => add_8, mul_10, mul_11, sub_13
#   input_7 => relu_2
#   mask_sum => full_default
#   mask_sum_2 => where_6
#   mul_3 => mul_12
#   new_mask_4 => full_default_10
#   new_mask_5 => where_8
#   no_update_holes_2 => eq_2
#   output_1 => full_default_1
#   output_4 => convolution_4
#   output_5 => where_7
#   output_pre_2 => add_6
#   sub_2 => sub_12
#   truediv_2 => div_2
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_8, %primals_17, %primals_18, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %eq_2 : [num_users=4] = call_function[target=torch.ops.aten.eq.Scalar](args = (%convolution_5, 0), kwargs = {})
#   %where_6 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_2, %full_default, %convolution_5), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %expand_2), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_12, %where_6), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, %expand_2), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_2, %full_default_1, %add_6), kwargs = {})
#   %full_default_10 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1, 1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%eq_2, %full_default_1, %full_default_10), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_7, %unsqueeze_17), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_19), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_21), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
#   %mul_12 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_2, %where_8), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_7, %unsqueeze_66), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i1', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 - tmp6
    tmp9 = tmp8 / tmp4
    tmp10 = tmp9 + tmp6
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = tmp11 - tmp12
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp18 / tmp17
    tmp20 = tmp19 * tmp3
    tmp21 = tmp13 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tmp28 = tl.where(tmp2, tmp1, tmp3)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp27 <= tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp29, xmask)
    tl.store(out_ptr5 + (x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gj/cgjrsz6tlfqzbtg55njx3dudomlc66lzbynnomaahi75agyopiy3.py
# Topologically Sorted Source Nodes: [mask_sum, output_1, output_6, no_update_holes_3, mask_sum_3, sub_3, truediv_3, output_pre_3, output_7, new_mask_6, new_mask_7, input_8, input_9, mul_4], Original ATen: [aten.masked_fill, aten.convolution, aten.eq, aten.sub, aten.div, aten.add, aten.ones_like, aten._native_batch_norm_legit_no_training, aten.relu, aten.mul, aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_8 => add_11, mul_14, mul_15, sub_15
#   input_9 => relu_3
#   mask_sum => full_default
#   mask_sum_3 => where_9
#   mul_4 => mul_16
#   new_mask_6 => full_default_14
#   new_mask_7 => where_11
#   no_update_holes_3 => eq_3
#   output_1 => full_default_1
#   output_6 => convolution_6
#   output_7 => where_10
#   output_pre_3 => add_9
#   sub_3 => sub_14
#   truediv_3 => div_3
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %primals_24, %primals_25, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %eq_3 : [num_users=4] = call_function[target=torch.ops.aten.eq.Scalar](args = (%convolution_7, 0), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_3, %full_default, %convolution_7), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %expand_3), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_14, %where_9), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %expand_3), kwargs = {})
#   %where_10 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_3, %full_default_1, %add_9), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 256, 1, 1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_11 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%eq_3, %full_default_1, %full_default_14), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_10, %unsqueeze_25), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_27), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %unsqueeze_29), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
#   %mul_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_3, %where_11), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_10, %unsqueeze_54), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i1', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 - tmp6
    tmp9 = tmp8 / tmp4
    tmp10 = tmp9 + tmp6
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = tmp11 - tmp12
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp18 / tmp17
    tmp20 = tmp19 * tmp3
    tmp21 = tmp13 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tmp28 = tl.where(tmp2, tmp1, tmp3)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp27 <= tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp29, xmask)
    tl.store(out_ptr5 + (x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/py/cpy3c2wqobzzrmzvnwhmnzw66ix3h3ie6ljjyk6zdljmijddxa2s.py
# Topologically Sorted Source Nodes: [mask_sum, output_1, output_8, no_update_holes_4, mask_sum_4, sub_4, truediv_4, output_pre_4, output_9, input_10, input_11], Original ATen: [aten.masked_fill, aten.convolution, aten.eq, aten.sub, aten.div, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_10 => add_14, mul_18, mul_19, sub_17
#   input_11 => relu_4
#   mask_sum => full_default
#   mask_sum_4 => where_12
#   no_update_holes_4 => eq_4
#   output_1 => full_default_1
#   output_8 => convolution_8
#   output_9 => where_13
#   output_pre_4 => add_12
#   sub_4 => sub_16
#   truediv_4 => div_4
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_16, %primals_31, %primals_32, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %eq_4 : [num_users=3] = call_function[target=torch.ops.aten.eq.Scalar](args = (%convolution_9, 0), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_4, %full_default, %convolution_9), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %expand_4), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_16, %where_12), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %expand_4), kwargs = {})
#   %where_13 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_4, %full_default_1, %add_12), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_13, %unsqueeze_33), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_35), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %unsqueeze_37), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_14,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_4, 0), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_13, %unsqueeze_42), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_native_batch_norm_backward_relu_sub_threshold_backward_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_native_batch_norm_backward_relu_sub_threshold_backward_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_native_batch_norm_backward_relu_sub_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_native_batch_norm_backward_relu_sub_threshold_backward_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 - tmp6
    tmp9 = tmp8 / tmp4
    tmp10 = tmp9 + tmp6
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = tmp11 - tmp12
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp18 / tmp17
    tmp20 = tmp19 * tmp3
    tmp21 = tmp13 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tmp28 = tmp27 <= tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp27, xmask)
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (32, 4, 7, 7), (196, 49, 7, 1))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, 4, 7, 7), (196, 49, 7, 1))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_25, (256, ), (1, ))
    assert_size_stride(primals_26, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 4, 7, 7), (196, 1, 28, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_3, buf0, 128, 49, grid=grid(128, 49), stream=stream0)
        del primals_3
        buf1 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_10, buf1, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_10
        buf2 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_17, buf2, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_17
        buf3 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_24, buf3, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_24
        buf4 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_31, buf4, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_31
        buf5 = empty_strided_cuda((4, 4, 10, 10), (400, 1, 40, 4), torch.float32)
        buf8 = empty_strided_cuda((4, 4, 10, 10), (400, 1, 40, 4), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, mask, mul], Original ATen: [aten.reflection_pad2d, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reflection_pad2d_5.run(primals_1, primals_2, buf5, buf8, 16, 100, grid=grid(16, 100), stream=stream0)
        del primals_1
        del primals_2
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, buf0, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 32, 4, 4), (512, 1, 128, 32))
        buf9 = empty_strided_cuda((32, 4, 7, 7), (196, 1, 28, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mask, output_mask], Original ATen: [aten.reflection_pad2d, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_5, buf9, 128, 49, grid=grid(128, 49), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [mask, output_mask], Original ATen: [aten.reflection_pad2d, aten.convolution]
        buf10 = extern_kernels.convolution(buf8, buf9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 32, 4, 4), (512, 1, 128, 32))
        del buf8
        del buf9
        buf7 = buf6; del buf6  # reuse
        buf11 = empty_strided_cuda((4, 32, 4, 4), (512, 1, 128, 32), torch.float32)
        buf13 = empty_strided_cuda((4, 32, 4, 4), (512, 1, 128, 32), torch.float32)
        # Topologically Sorted Source Nodes: [output, no_update_holes, mask_sum, sub, truediv, output_pre, output_1, new_mask, new_mask_1, input_2, input_3, mul_1], Original ATen: [aten.convolution, aten.eq, aten.masked_fill, aten.sub, aten.div, aten.add, aten.ones_like, aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_6.run(buf7, primals_4, buf10, primals_6, primals_7, primals_8, primals_9, buf11, buf13, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 64, 2, 2), (256, 1, 128, 64))
        buf14 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [no_update_holes, output_1, new_mask, new_mask_1, output_mask_1], Original ATen: [aten.eq, aten.masked_fill, aten.ones_like, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_12, buf14, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_12
        # Topologically Sorted Source Nodes: [no_update_holes, output_1, new_mask, new_mask_1, output_mask_1], Original ATen: [aten.eq, aten.masked_fill, aten.ones_like, aten.convolution]
        buf15 = extern_kernels.convolution(buf13, buf14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 64, 2, 2), (256, 1, 128, 64))
        del buf14
        buf16 = empty_strided_cuda((4, 64, 2, 2), (256, 1, 128, 64), torch.bool)
        buf17 = buf15; del buf15  # reuse
        buf18 = buf12; del buf12  # reuse
        buf19 = empty_strided_cuda((4, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        buf20 = empty_strided_cuda((4, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Topologically Sorted Source Nodes: [mask_sum, output_1, output_2, no_update_holes_1, mask_sum_1, sub_1, truediv_1, output_pre_1, output_3, new_mask_2, new_mask_3, input_4, input_5, mul_2], Original ATen: [aten.masked_fill, aten.convolution, aten.eq, aten.sub, aten.div, aten.add, aten.ones_like, aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_ones_like_relu_sub_7.run(buf17, buf18, primals_11, primals_13, primals_14, primals_15, primals_16, buf16, buf19, buf20, 1024, grid=grid(1024), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [output_4], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 128, 1, 1), (128, 1, 128, 128))
        buf22 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [output_mask_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_19, buf22, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [output_mask_2], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf19, buf22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 128, 1, 1), (128, 1, 128, 128))
        del buf22
        buf24 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        buf25 = buf23; del buf23  # reuse
        buf48 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf26 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf28 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf47 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        # Topologically Sorted Source Nodes: [mask_sum, output_1, output_4, no_update_holes_2, mask_sum_2, sub_2, truediv_2, output_pre_2, output_5, new_mask_4, new_mask_5, input_6, input_7, mul_3], Original ATen: [aten.masked_fill, aten.convolution, aten.eq, aten.sub, aten.div, aten.add, aten.ones_like, aten._native_batch_norm_legit_no_training, aten.relu, aten.mul, aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_8.run(buf25, buf21, primals_18, primals_20, primals_21, primals_22, primals_23, buf24, buf48, buf26, buf28, buf47, 512, grid=grid(512), stream=stream0)
        del buf21
        del primals_18
        del primals_20
        del primals_23
        # Topologically Sorted Source Nodes: [output_6], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 256, 1, 1), (256, 1, 256, 256))
        buf30 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [output_mask_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_26, buf30, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [output_mask_3], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf26, buf30, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 256, 1, 1), (256, 1, 256, 256))
        del buf30
        buf32 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.bool)
        buf33 = buf31; del buf31  # reuse
        buf46 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf34 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf36 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf45 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.bool)
        # Topologically Sorted Source Nodes: [mask_sum, output_1, output_6, no_update_holes_3, mask_sum_3, sub_3, truediv_3, output_pre_3, output_7, new_mask_6, new_mask_7, input_8, input_9, mul_4], Original ATen: [aten.masked_fill, aten.convolution, aten.eq, aten.sub, aten.div, aten.add, aten.ones_like, aten._native_batch_norm_legit_no_training, aten.relu, aten.mul, aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_mul_native_batch_norm_backward_ones_like_relu_sub_threshold_backward_9.run(buf33, buf29, primals_25, primals_27, primals_28, primals_29, primals_30, buf32, buf46, buf34, buf36, buf45, 1024, grid=grid(1024), stream=stream0)
        del buf29
        del primals_25
        del primals_27
        del primals_30
        # Topologically Sorted Source Nodes: [output_8], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 512, 1, 1), (512, 1, 512, 512))
        buf38 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [output_mask_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_33, buf38, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [output_mask_4], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf34, buf38, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 512, 1, 1), (512, 1, 512, 512))
        del buf38
        buf40 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf41 = buf39; del buf39  # reuse
        buf42 = reinterpret_tensor(buf13, (4, 512, 1, 1), (512, 1, 1, 1), 0); del buf13  # reuse
        buf44 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf43 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        # Topologically Sorted Source Nodes: [mask_sum, output_1, output_8, no_update_holes_4, mask_sum_4, sub_4, truediv_4, output_pre_4, output_9, input_10, input_11], Original ATen: [aten.masked_fill, aten.convolution, aten.eq, aten.sub, aten.div, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_eq_masked_fill_native_batch_norm_backward_relu_sub_threshold_backward_10.run(buf41, buf37, primals_32, primals_34, primals_35, primals_36, primals_37, buf40, buf42, buf44, buf43, 2048, grid=grid(2048), stream=stream0)
        del buf37
        del primals_32
        del primals_34
        del primals_37
    return (buf42, buf0, primals_4, primals_6, primals_7, primals_8, primals_9, buf1, primals_13, primals_14, primals_15, primals_16, buf2, primals_21, primals_22, buf3, primals_28, primals_29, buf4, primals_35, primals_36, buf5, buf7, buf10, buf11, buf16, buf17, buf18, buf19, buf20, buf24, buf25, buf26, buf28, buf32, buf33, buf34, buf36, buf40, buf41, buf43, buf44, buf45, buf46, buf47, buf48, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 4, 7, 7), (196, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, 4, 7, 7), (196, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
