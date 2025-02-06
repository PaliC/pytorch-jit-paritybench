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


# kernel path: inductor_cache/xk/cxkaxztltouhwtswddb6rnpxa5oonqovm7g3iqjgkrqbwvicrh3q.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 72
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


# kernel path: inductor_cache/vf/cvfeesrttcyslzaczrad3hru6qd5a7tjb6icahjelpvqht3gqeth.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/jo/cjop2jscigpfdowm6npo37i2rfc5eaf2oaor2aglg2zbic6l2425.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
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


# kernel path: inductor_cache/lb/clbvcn2ovmhqrtkk5antxevqrobjxs2f7aspjgknsyj34u56qcih.py
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
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 48)
    y1 = yindex // 48
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 48*x2 + 432*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j3/cj3aqi2t7tn2uazy6pjodhxzr2ba6kojtzy2vgz6pgsu23idndl2.py
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
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nj/cnjjbz5jh5iplszzbcrhquq5nwaqptwsyqkkktixh37qckdiopr3.py
# Topologically Sorted Source Nodes: [mul, x, sub, x_1], Original ATen: [aten.mul, aten.add, aten.sub, aten.div]
# Source node to ATen node mapping:
#   mul => mul
#   sub => sub
#   x => add
#   x_1 => div
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, 0.5), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0.5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %view), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %view_1), kwargs = {})
triton_poi_fused_add_div_mul_sub_5 = async_compile.triton('triton_poi_fused_add_div_mul_sub_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sub_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_sub_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr0 + (y0 + 3*x2 + 48*y1), tmp7, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/cq/ccqbvuk55253q4m25cu2itjsfxe2jerxwfexpimm3wmyjmdd57dh.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   input_2 => add_2, mul_2, mul_3, sub_1
#   input_3 => mul_4, sigmoid
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_3), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %unsqueeze_5), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %unsqueeze_7), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_4 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4j/c4j3epc7hl3is6jxpgocmdepo5o6vlindadfzdu4zajkkn6zryxb.py
# Topologically Sorted Source Nodes: [input_5, input_6, result], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
# Source node to ATen node mapping:
#   input_5 => add_4, mul_6, mul_7, sub_2
#   input_6 => mul_8, sigmoid_1
#   result => add_5
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_11), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_13), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_15), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_4,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %sigmoid_1), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_4), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sy/csyqmvwjuf46fja3mpgmhi7xazn6trnblou2slxi4t5qw3g7jsdc.py
# Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   input_11 => add_10, mul_14, mul_15, sub_4
#   input_12 => mul_16, sigmoid_3
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_27), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %unsqueeze_29), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %unsqueeze_31), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_10,), kwargs = {})
#   %mul_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %sigmoid_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ro/cromatq2hlzc5yjmsuaijaehkiwimffrd27zoxy57uh2ud4xi5bp.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_14 => add_12, mul_18, mul_19, sub_5
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_35), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %unsqueeze_37), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/75/c7536oto3zhcaepqs76fomyhuvo3ta3wfmtd2yfcvtiqzhygmqa4.py
# Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   input_16 => add_14, mul_21, mul_22, sub_6
#   input_17 => mul_23, sigmoid_4
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_43), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_47), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_14,), kwargs = {})
#   %mul_23 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %sigmoid_4), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ir/cirrqjnpznsxoaupxjpqs6kwlqtyic2jlvw3vufcce4fbo3lycco.py
# Topologically Sorted Source Nodes: [input_19, result_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_19 => add_16, mul_25, mul_26, sub_7
#   result_2 => add_17
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_51), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_53), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_55), kwargs = {})
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %add_12), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5x/c5xcvgybus4hcpow6oqtw6bptdfq4pkgvlzptes7wlqkit2hjsu4.py
# Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_34 => add_31, mul_46, mul_47, sub_13
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_99), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_101), kwargs = {})
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_103), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bi/cbiei6yznz6uey3so7u5xy2l2djxlpddw6mz5s2rzsycsafapvhs.py
# Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   input_36 => add_33, mul_49, mul_50, sub_14
#   input_37 => mul_51, sigmoid_8
# Graph fragment:
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_107), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_109), kwargs = {})
#   %add_33 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_111), kwargs = {})
#   %sigmoid_8 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_33,), kwargs = {})
#   %mul_51 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_33, %sigmoid_8), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6t/c6tsmkjdo4uuhbzdxn2wqtn4zuna7ok7qvn2g5sgvy67asqcwqrr.py
# Topologically Sorted Source Nodes: [input_39, result_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_39 => add_35, mul_53, mul_54, sub_15
#   result_5 => add_36
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_115), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, %unsqueeze_117), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, %unsqueeze_119), kwargs = {})
#   %add_36 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %add_31), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eh/ceh54zh6bpjocjzpmlonuqz24yfogfbpoqnvkdbgq73uyol7jba7.py
# Topologically Sorted Source Nodes: [input_54, input_55, scale], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   input_54 => add_50, mul_74, mul_75, sub_21
#   input_55 => mul_76, sigmoid_12
#   scale => mean
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_163), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_74, %unsqueeze_165), kwargs = {})
#   %add_50 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %unsqueeze_167), kwargs = {})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_50,), kwargs = {})
#   %mul_76 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_50, %sigmoid_12), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_76, [-1, -2], True), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 / tmp9
    tl.store(out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr1 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tf/ctfpcnmdaozz2prfonp26gzy5czajapue5a5pe5vyhgqjlv46nix.py
# Topologically Sorted Source Nodes: [scale_1, scale_2], Original ATen: [aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   scale_1 => convolution_21
#   scale_2 => mul_77, sigmoid_13
# Graph fragment:
#   %convolution_21 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mean, %primals_109, %primals_110, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_21,), kwargs = {})
#   %mul_77 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_21, %sigmoid_13), kwargs = {})
triton_poi_fused_convolution_silu_16 = async_compile.triton('triton_poi_fused_convolution_silu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_16(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e6/ce6lyqlqn3sdlmpys5n32k73x6vsvafejgstfkjqygne7zltiszk.py
# Topologically Sorted Source Nodes: [input_55, scale_3, scale_4, input_56], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_55 => mul_76, sigmoid_12
#   input_56 => mul_78
#   scale_3 => convolution_22
#   scale_4 => sigmoid_14
# Graph fragment:
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_50,), kwargs = {})
#   %mul_76 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_50, %sigmoid_12), kwargs = {})
#   %convolution_22 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_77, %primals_111, %primals_112, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_22,), kwargs = {})
#   %mul_78 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_14, %mul_76), kwargs = {})
triton_poi_fused_convolution_mul_sigmoid_silu_17 = async_compile.triton('triton_poi_fused_convolution_mul_sigmoid_silu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sigmoid_silu_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_silu_17(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 * tmp6
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rr/crrs6orlxf2yblgaeobx75affn6l3kbzgiejcbhwlxg3vtgvr7r3.py
# Topologically Sorted Source Nodes: [input_58], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_58 => add_52, mul_80, mul_81, sub_22
# Graph fragment:
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_169), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_171), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_80, %unsqueeze_173), kwargs = {})
#   %add_52 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %unsqueeze_175), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sx/csxelixzse33cahm7dssgg4k2ijrhynnjaqtvfxb2w6etu6aqsg4.py
# Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   input_60 => add_54, mul_83, mul_84, sub_23
#   input_61 => mul_85, sigmoid_15
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_177), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_179), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, %unsqueeze_181), kwargs = {})
#   %add_54 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %unsqueeze_183), kwargs = {})
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_54,), kwargs = {})
#   %mul_85 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, %sigmoid_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xv/cxvp244xzskz6egcnhvzqy4imgi2bkl6tuq3n2h7sdww7h2owd54.py
# Topologically Sorted Source Nodes: [input_63, input_64, scale_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   input_63 => add_56, mul_87, mul_88, sub_24
#   input_64 => mul_89, sigmoid_16
#   scale_5 => mean_1
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_185), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_187), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %unsqueeze_189), kwargs = {})
#   %add_56 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %unsqueeze_191), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_56,), kwargs = {})
#   %mul_89 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_56, %sigmoid_16), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_89, [-1, -2], True), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 / tmp9
    tl.store(out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr1 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2b/c2balje5yzdtoxi5mfyv5cnsivkb62degdjukx6ak7s4dqdldxex.py
# Topologically Sorted Source Nodes: [scale_6, scale_7], Original ATen: [aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   scale_6 => convolution_26
#   scale_7 => mul_90, sigmoid_17
# Graph fragment:
#   %convolution_26 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_1, %primals_128, %primals_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_26,), kwargs = {})
#   %mul_90 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_26, %sigmoid_17), kwargs = {})
triton_poi_fused_convolution_silu_21 = async_compile.triton('triton_poi_fused_convolution_silu_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_21(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ce/cceyum7b4hlwy3vnu6hqtnmxjnqgldqixsgsy2ukhowhs3g56e2n.py
# Topologically Sorted Source Nodes: [input_64, scale_8, scale_9, input_65], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_64 => mul_89, sigmoid_16
#   input_65 => mul_91
#   scale_8 => convolution_27
#   scale_9 => sigmoid_18
# Graph fragment:
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_56,), kwargs = {})
#   %mul_89 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_56, %sigmoid_16), kwargs = {})
#   %convolution_27 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_90, %primals_130, %primals_131, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_18 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_27,), kwargs = {})
#   %mul_91 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_18, %mul_89), kwargs = {})
triton_poi_fused_convolution_mul_sigmoid_silu_22 = async_compile.triton('triton_poi_fused_convolution_mul_sigmoid_silu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sigmoid_silu_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_silu_22(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 * tmp6
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oc/cocm7zsvk6zldfn5roeyotih7xresn3jkuypi5g6egzdwqa6jwbk.py
# Topologically Sorted Source Nodes: [input_67, result_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_67 => add_58, mul_93, mul_94, sub_25
#   result_8 => add_59
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_193), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_195), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %unsqueeze_197), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %unsqueeze_199), kwargs = {})
#   %add_59 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_58, %add_52), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u6/cu6wyry3xc3yq3ludt2b44wn752wsosf2fl46es2sidlklbiuaeo.py
# Topologically Sorted Source Nodes: [input_105, input_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   input_105 => add_89, mul_148, mul_149, sub_38
#   input_106 => mul_150, sigmoid_35
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_297), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_299), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_301), kwargs = {})
#   %add_89 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_303), kwargs = {})
#   %sigmoid_35 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_150 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_35), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 768)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/35/c357oharqgbeskm6cxcddk4xmpq2zjjnfsirs6jzr5argg4nkf4s.py
# Topologically Sorted Source Nodes: [input_108, input_109, scale_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   input_108 => add_91, mul_152, mul_153, sub_39
#   input_109 => mul_154, sigmoid_36
#   scale_30 => mean_6
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_305), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_307), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_152, %unsqueeze_309), kwargs = {})
#   %add_91 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_153, %unsqueeze_311), kwargs = {})
#   %sigmoid_36 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_91,), kwargs = {})
#   %mul_154 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_91, %sigmoid_36), kwargs = {})
#   %mean_6 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_154, [-1, -2], True), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 768)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 / tmp9
    tl.store(out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr1 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zy/czykqayii35wsko64gir3ni37umfxjh23wthwvxhpdz2fgzfdhx7.py
# Topologically Sorted Source Nodes: [input_109, scale_33, scale_34, input_110], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_109 => mul_154, sigmoid_36
#   input_110 => mul_156
#   scale_33 => convolution_52
#   scale_34 => sigmoid_38
# Graph fragment:
#   %sigmoid_36 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_91,), kwargs = {})
#   %mul_154 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_91, %sigmoid_36), kwargs = {})
#   %convolution_52 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_155, %primals_225, %primals_226, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_38 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_52,), kwargs = {})
#   %mul_156 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_38, %mul_154), kwargs = {})
triton_poi_fused_convolution_mul_sigmoid_silu_26 = async_compile.triton('triton_poi_fused_convolution_mul_sigmoid_silu_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sigmoid_silu_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_silu_26(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 * tmp6
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fk/cfknnpl7ewl2jqu33bpgmbwnj33dweusqx3dqot3unqf723micmc.py
# Topologically Sorted Source Nodes: [input_112], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_112 => add_93, mul_158, mul_159, sub_40
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_313), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_315), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_158, %unsqueeze_317), kwargs = {})
#   %add_93 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_159, %unsqueeze_319), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ig/cigwaidxaler57pcdglns3s4s77auofrhazw2cezxhuij2u6aezb.py
# Topologically Sorted Source Nodes: [input_114, input_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   input_114 => add_95, mul_161, mul_162, sub_41
#   input_115 => mul_163, sigmoid_39
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_54, %unsqueeze_321), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_323), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_161, %unsqueeze_325), kwargs = {})
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_162, %unsqueeze_327), kwargs = {})
#   %sigmoid_39 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_95,), kwargs = {})
#   %mul_163 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_95, %sigmoid_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 960)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sj/csjxhlkd445plkxlsahdg7c73nfeozbgovcr3j5wrwqjie7wo4wl.py
# Topologically Sorted Source Nodes: [input_117, input_118, scale_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   input_117 => add_97, mul_165, mul_166, sub_42
#   input_118 => mul_167, sigmoid_40
#   scale_35 => mean_7
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_55, %unsqueeze_329), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_331), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_165, %unsqueeze_333), kwargs = {})
#   %add_97 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_166, %unsqueeze_335), kwargs = {})
#   %sigmoid_40 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_97,), kwargs = {})
#   %mul_167 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_97, %sigmoid_40), kwargs = {})
#   %mean_7 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_167, [-1, -2], True), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 960)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 / tmp9
    tl.store(out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr1 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m6/cm6mxif3kfgiouqkm6hf5eurbptlfsmtloadvqkpz4ubipz3r2xk.py
# Topologically Sorted Source Nodes: [scale_36, scale_37], Original ATen: [aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   scale_36 => convolution_56
#   scale_37 => mul_168, sigmoid_41
# Graph fragment:
#   %convolution_56 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_7, %primals_242, %primals_243, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_41 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_56,), kwargs = {})
#   %mul_168 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_56, %sigmoid_41), kwargs = {})
triton_poi_fused_convolution_silu_30 = async_compile.triton('triton_poi_fused_convolution_silu_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_30(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qj/cqjahjkb7c6vbwem6sfgqsjeq6cmqeymesqv6jqenu2ek3uawkkz.py
# Topologically Sorted Source Nodes: [input_118, scale_38, scale_39, input_119], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_118 => mul_167, sigmoid_40
#   input_119 => mul_169
#   scale_38 => convolution_57
#   scale_39 => sigmoid_42
# Graph fragment:
#   %sigmoid_40 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_97,), kwargs = {})
#   %mul_167 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_97, %sigmoid_40), kwargs = {})
#   %convolution_57 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_168, %primals_244, %primals_245, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_42 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_57,), kwargs = {})
#   %mul_169 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_42, %mul_167), kwargs = {})
triton_poi_fused_convolution_mul_sigmoid_silu_31 = async_compile.triton('triton_poi_fused_convolution_mul_sigmoid_silu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sigmoid_silu_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_silu_31(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 960)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 * tmp6
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/db/cdbbl2gcjriwfj4f57oxvdvoggigziohc4y3ytb5u4p2nq4y56qv.py
# Topologically Sorted Source Nodes: [input_121, result_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_121 => add_99, mul_171, mul_172, sub_43
#   result_13 => add_100
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_337), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_339), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_171, %unsqueeze_341), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_172, %unsqueeze_343), kwargs = {})
#   %add_100 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_99, %add_93), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640
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
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lr/clrdwnho6woor2oeajeluhhwk3hzwuszmrkso33ynflpqmzjkp6w.py
# Topologically Sorted Source Nodes: [input_193], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_193 => add_155, mul_275, mul_276, sub_67
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_98, %unsqueeze_529), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_531), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_275, %unsqueeze_533), kwargs = {})
#   %add_155 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_276, %unsqueeze_535), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cp/ccp7ih4ocn65kumti46wexwuo5rihy2wd4rymcm7rt2532l77skh.py
# Topologically Sorted Source Nodes: [input_195, input_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   input_195 => add_157, mul_278, mul_279, sub_68
#   input_196 => mul_280, sigmoid_75
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_99, %unsqueeze_537), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_539), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_278, %unsqueeze_541), kwargs = {})
#   %add_157 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_279, %unsqueeze_543), kwargs = {})
#   %sigmoid_75 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_157,), kwargs = {})
#   %mul_280 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_157, %sigmoid_75), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1536)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7j/c7jmar5476b6tudaf65okvypgwdcki3225bie2opf4dhbinv3x2e.py
# Topologically Sorted Source Nodes: [input_198, input_199, scale_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   input_198 => add_159, mul_282, mul_283, sub_69
#   input_199 => mul_284, sigmoid_76
#   scale_80 => mean_16
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_545), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_547), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_282, %unsqueeze_549), kwargs = {})
#   %add_159 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_283, %unsqueeze_551), kwargs = {})
#   %sigmoid_76 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_159,), kwargs = {})
#   %mul_284 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_159, %sigmoid_76), kwargs = {})
#   %mean_16 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_284, [-1, -2], True), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1536)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 / tmp9
    tl.store(out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr1 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7d/c7dulha6clhcz3bmrqjcyghh6k5gzwven6idphqbdbfxgpk5kmoo.py
# Topologically Sorted Source Nodes: [scale_81, scale_82], Original ATen: [aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   scale_81 => convolution_101
#   scale_82 => mul_285, sigmoid_77
# Graph fragment:
#   %convolution_101 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %primals_413, %primals_414, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_77 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_101,), kwargs = {})
#   %mul_285 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_101, %sigmoid_77), kwargs = {})
triton_poi_fused_convolution_silu_36 = async_compile.triton('triton_poi_fused_convolution_silu_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_36(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jw/cjwckazjiqagr4rlcsyu2v7geiugzn6u2vvntbe4icilx6lcwbp5.py
# Topologically Sorted Source Nodes: [input_199, scale_83, scale_84, input_200], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_199 => mul_284, sigmoid_76
#   input_200 => mul_286
#   scale_83 => convolution_102
#   scale_84 => sigmoid_78
# Graph fragment:
#   %sigmoid_76 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_159,), kwargs = {})
#   %mul_284 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_159, %sigmoid_76), kwargs = {})
#   %convolution_102 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_285, %primals_415, %primals_416, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_102,), kwargs = {})
#   %mul_286 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_78, %mul_284), kwargs = {})
triton_poi_fused_convolution_mul_sigmoid_silu_37 = async_compile.triton('triton_poi_fused_convolution_mul_sigmoid_silu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sigmoid_silu_37', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_silu_37(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1536)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 * tmp6
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lj/cljc64pq3sroq4sxh2ukmmoy7zkjx6faysrrcrwm7mdqgzksqv7y.py
# Topologically Sorted Source Nodes: [input_202, result_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_202 => add_161, mul_288, mul_289, sub_70
#   result_21 => add_162
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_553), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_555), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_288, %unsqueeze_557), kwargs = {})
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_289, %unsqueeze_559), kwargs = {})
#   %add_162 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_161, %add_155), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/na/cna5mig4khkrwdpm4bqec47dbwswlvmafdoskkc6jeha3xwf7ifw.py
# Topologically Sorted Source Nodes: [input_321, input_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   input_321 => add_255, mul_460, mul_461, sub_110
#   input_322 => mul_462, sigmoid_131
# Graph fragment:
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_169, %unsqueeze_873), kwargs = {})
#   %mul_460 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %unsqueeze_875), kwargs = {})
#   %mul_461 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_460, %unsqueeze_877), kwargs = {})
#   %add_255 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_461, %unsqueeze_879), kwargs = {})
#   %sigmoid_131 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_255,), kwargs = {})
#   %mul_462 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_255, %sigmoid_131), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1280)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/iu/ciuekavx7k64gf6rfh237nj5w75qtpjbitck5ju6734bqfwpiget.py
# Topologically Sorted Source Nodes: [input_324], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_324 => mul_464, sub_111
# Graph fragment:
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_170, %unsqueeze_881), kwargs = {})
#   %mul_464 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_883), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_40(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp11, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_2, (3, ), (1, ))
    assert_size_stride(primals_3, (3, ), (1, ))
    assert_size_stride(primals_4, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (24, ), (1, ))
    assert_size_stride(primals_7, (24, ), (1, ))
    assert_size_stride(primals_8, (24, ), (1, ))
    assert_size_stride(primals_9, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_10, (24, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_12, (24, ), (1, ))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_14, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_15, (24, ), (1, ))
    assert_size_stride(primals_16, (24, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (96, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_20, (96, ), (1, ))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_22, (96, ), (1, ))
    assert_size_stride(primals_23, (96, ), (1, ))
    assert_size_stride(primals_24, (48, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_25, (48, ), (1, ))
    assert_size_stride(primals_26, (48, ), (1, ))
    assert_size_stride(primals_27, (48, ), (1, ))
    assert_size_stride(primals_28, (48, ), (1, ))
    assert_size_stride(primals_29, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_30, (192, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_35, (48, ), (1, ))
    assert_size_stride(primals_36, (48, ), (1, ))
    assert_size_stride(primals_37, (48, ), (1, ))
    assert_size_stride(primals_38, (48, ), (1, ))
    assert_size_stride(primals_39, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (192, ), (1, ))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_44, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_45, (48, ), (1, ))
    assert_size_stride(primals_46, (48, ), (1, ))
    assert_size_stride(primals_47, (48, ), (1, ))
    assert_size_stride(primals_48, (48, ), (1, ))
    assert_size_stride(primals_49, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_50, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_52, (192, ), (1, ))
    assert_size_stride(primals_53, (192, ), (1, ))
    assert_size_stride(primals_54, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_55, (48, ), (1, ))
    assert_size_stride(primals_56, (48, ), (1, ))
    assert_size_stride(primals_57, (48, ), (1, ))
    assert_size_stride(primals_58, (48, ), (1, ))
    assert_size_stride(primals_59, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_62, (192, ), (1, ))
    assert_size_stride(primals_63, (192, ), (1, ))
    assert_size_stride(primals_64, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_82, (256, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (64, ), (1, ))
    assert_size_stride(primals_87, (64, ), (1, ))
    assert_size_stride(primals_88, (64, ), (1, ))
    assert_size_stride(primals_89, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (256, ), (1, ))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_96, (64, ), (1, ))
    assert_size_stride(primals_97, (64, ), (1, ))
    assert_size_stride(primals_98, (64, ), (1, ))
    assert_size_stride(primals_99, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_104, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (16, ), (1, ))
    assert_size_stride(primals_111, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (128, ), (1, ))
    assert_size_stride(primals_118, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_123, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (32, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_129, (32, ), (1, ))
    assert_size_stride(primals_130, (512, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_137, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_138, (512, ), (1, ))
    assert_size_stride(primals_139, (512, ), (1, ))
    assert_size_stride(primals_140, (512, ), (1, ))
    assert_size_stride(primals_141, (512, ), (1, ))
    assert_size_stride(primals_142, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, ), (1, ))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (32, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_148, (32, ), (1, ))
    assert_size_stride(primals_149, (512, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_150, (512, ), (1, ))
    assert_size_stride(primals_151, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_157, (512, ), (1, ))
    assert_size_stride(primals_158, (512, ), (1, ))
    assert_size_stride(primals_159, (512, ), (1, ))
    assert_size_stride(primals_160, (512, ), (1, ))
    assert_size_stride(primals_161, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_162, (512, ), (1, ))
    assert_size_stride(primals_163, (512, ), (1, ))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (32, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_167, (32, ), (1, ))
    assert_size_stride(primals_168, (512, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_169, (512, ), (1, ))
    assert_size_stride(primals_170, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_171, (128, ), (1, ))
    assert_size_stride(primals_172, (128, ), (1, ))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_176, (512, ), (1, ))
    assert_size_stride(primals_177, (512, ), (1, ))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_181, (512, ), (1, ))
    assert_size_stride(primals_182, (512, ), (1, ))
    assert_size_stride(primals_183, (512, ), (1, ))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (32, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_186, (32, ), (1, ))
    assert_size_stride(primals_187, (512, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_188, (512, ), (1, ))
    assert_size_stride(primals_189, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_190, (128, ), (1, ))
    assert_size_stride(primals_191, (128, ), (1, ))
    assert_size_stride(primals_192, (128, ), (1, ))
    assert_size_stride(primals_193, (128, ), (1, ))
    assert_size_stride(primals_194, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (512, ), (1, ))
    assert_size_stride(primals_198, (512, ), (1, ))
    assert_size_stride(primals_199, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_200, (512, ), (1, ))
    assert_size_stride(primals_201, (512, ), (1, ))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (32, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_205, (32, ), (1, ))
    assert_size_stride(primals_206, (512, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_207, (512, ), (1, ))
    assert_size_stride(primals_208, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_209, (128, ), (1, ))
    assert_size_stride(primals_210, (128, ), (1, ))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (768, ), (1, ))
    assert_size_stride(primals_218, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_219, (768, ), (1, ))
    assert_size_stride(primals_220, (768, ), (1, ))
    assert_size_stride(primals_221, (768, ), (1, ))
    assert_size_stride(primals_222, (768, ), (1, ))
    assert_size_stride(primals_223, (32, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_224, (32, ), (1, ))
    assert_size_stride(primals_225, (768, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_226, (768, ), (1, ))
    assert_size_stride(primals_227, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_228, (160, ), (1, ))
    assert_size_stride(primals_229, (160, ), (1, ))
    assert_size_stride(primals_230, (160, ), (1, ))
    assert_size_stride(primals_231, (160, ), (1, ))
    assert_size_stride(primals_232, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_233, (960, ), (1, ))
    assert_size_stride(primals_234, (960, ), (1, ))
    assert_size_stride(primals_235, (960, ), (1, ))
    assert_size_stride(primals_236, (960, ), (1, ))
    assert_size_stride(primals_237, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_238, (960, ), (1, ))
    assert_size_stride(primals_239, (960, ), (1, ))
    assert_size_stride(primals_240, (960, ), (1, ))
    assert_size_stride(primals_241, (960, ), (1, ))
    assert_size_stride(primals_242, (40, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_243, (40, ), (1, ))
    assert_size_stride(primals_244, (960, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_245, (960, ), (1, ))
    assert_size_stride(primals_246, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_247, (160, ), (1, ))
    assert_size_stride(primals_248, (160, ), (1, ))
    assert_size_stride(primals_249, (160, ), (1, ))
    assert_size_stride(primals_250, (160, ), (1, ))
    assert_size_stride(primals_251, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_252, (960, ), (1, ))
    assert_size_stride(primals_253, (960, ), (1, ))
    assert_size_stride(primals_254, (960, ), (1, ))
    assert_size_stride(primals_255, (960, ), (1, ))
    assert_size_stride(primals_256, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_257, (960, ), (1, ))
    assert_size_stride(primals_258, (960, ), (1, ))
    assert_size_stride(primals_259, (960, ), (1, ))
    assert_size_stride(primals_260, (960, ), (1, ))
    assert_size_stride(primals_261, (40, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_262, (40, ), (1, ))
    assert_size_stride(primals_263, (960, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_264, (960, ), (1, ))
    assert_size_stride(primals_265, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_266, (160, ), (1, ))
    assert_size_stride(primals_267, (160, ), (1, ))
    assert_size_stride(primals_268, (160, ), (1, ))
    assert_size_stride(primals_269, (160, ), (1, ))
    assert_size_stride(primals_270, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_271, (960, ), (1, ))
    assert_size_stride(primals_272, (960, ), (1, ))
    assert_size_stride(primals_273, (960, ), (1, ))
    assert_size_stride(primals_274, (960, ), (1, ))
    assert_size_stride(primals_275, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_276, (960, ), (1, ))
    assert_size_stride(primals_277, (960, ), (1, ))
    assert_size_stride(primals_278, (960, ), (1, ))
    assert_size_stride(primals_279, (960, ), (1, ))
    assert_size_stride(primals_280, (40, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_281, (40, ), (1, ))
    assert_size_stride(primals_282, (960, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_283, (960, ), (1, ))
    assert_size_stride(primals_284, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_285, (160, ), (1, ))
    assert_size_stride(primals_286, (160, ), (1, ))
    assert_size_stride(primals_287, (160, ), (1, ))
    assert_size_stride(primals_288, (160, ), (1, ))
    assert_size_stride(primals_289, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_290, (960, ), (1, ))
    assert_size_stride(primals_291, (960, ), (1, ))
    assert_size_stride(primals_292, (960, ), (1, ))
    assert_size_stride(primals_293, (960, ), (1, ))
    assert_size_stride(primals_294, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_295, (960, ), (1, ))
    assert_size_stride(primals_296, (960, ), (1, ))
    assert_size_stride(primals_297, (960, ), (1, ))
    assert_size_stride(primals_298, (960, ), (1, ))
    assert_size_stride(primals_299, (40, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_300, (40, ), (1, ))
    assert_size_stride(primals_301, (960, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_302, (960, ), (1, ))
    assert_size_stride(primals_303, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_304, (160, ), (1, ))
    assert_size_stride(primals_305, (160, ), (1, ))
    assert_size_stride(primals_306, (160, ), (1, ))
    assert_size_stride(primals_307, (160, ), (1, ))
    assert_size_stride(primals_308, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_309, (960, ), (1, ))
    assert_size_stride(primals_310, (960, ), (1, ))
    assert_size_stride(primals_311, (960, ), (1, ))
    assert_size_stride(primals_312, (960, ), (1, ))
    assert_size_stride(primals_313, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_314, (960, ), (1, ))
    assert_size_stride(primals_315, (960, ), (1, ))
    assert_size_stride(primals_316, (960, ), (1, ))
    assert_size_stride(primals_317, (960, ), (1, ))
    assert_size_stride(primals_318, (40, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_319, (40, ), (1, ))
    assert_size_stride(primals_320, (960, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_321, (960, ), (1, ))
    assert_size_stride(primals_322, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_323, (160, ), (1, ))
    assert_size_stride(primals_324, (160, ), (1, ))
    assert_size_stride(primals_325, (160, ), (1, ))
    assert_size_stride(primals_326, (160, ), (1, ))
    assert_size_stride(primals_327, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_328, (960, ), (1, ))
    assert_size_stride(primals_329, (960, ), (1, ))
    assert_size_stride(primals_330, (960, ), (1, ))
    assert_size_stride(primals_331, (960, ), (1, ))
    assert_size_stride(primals_332, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_333, (960, ), (1, ))
    assert_size_stride(primals_334, (960, ), (1, ))
    assert_size_stride(primals_335, (960, ), (1, ))
    assert_size_stride(primals_336, (960, ), (1, ))
    assert_size_stride(primals_337, (40, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_338, (40, ), (1, ))
    assert_size_stride(primals_339, (960, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_340, (960, ), (1, ))
    assert_size_stride(primals_341, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_342, (160, ), (1, ))
    assert_size_stride(primals_343, (160, ), (1, ))
    assert_size_stride(primals_344, (160, ), (1, ))
    assert_size_stride(primals_345, (160, ), (1, ))
    assert_size_stride(primals_346, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_347, (960, ), (1, ))
    assert_size_stride(primals_348, (960, ), (1, ))
    assert_size_stride(primals_349, (960, ), (1, ))
    assert_size_stride(primals_350, (960, ), (1, ))
    assert_size_stride(primals_351, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_352, (960, ), (1, ))
    assert_size_stride(primals_353, (960, ), (1, ))
    assert_size_stride(primals_354, (960, ), (1, ))
    assert_size_stride(primals_355, (960, ), (1, ))
    assert_size_stride(primals_356, (40, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_357, (40, ), (1, ))
    assert_size_stride(primals_358, (960, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_359, (960, ), (1, ))
    assert_size_stride(primals_360, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_361, (160, ), (1, ))
    assert_size_stride(primals_362, (160, ), (1, ))
    assert_size_stride(primals_363, (160, ), (1, ))
    assert_size_stride(primals_364, (160, ), (1, ))
    assert_size_stride(primals_365, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_366, (960, ), (1, ))
    assert_size_stride(primals_367, (960, ), (1, ))
    assert_size_stride(primals_368, (960, ), (1, ))
    assert_size_stride(primals_369, (960, ), (1, ))
    assert_size_stride(primals_370, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_371, (960, ), (1, ))
    assert_size_stride(primals_372, (960, ), (1, ))
    assert_size_stride(primals_373, (960, ), (1, ))
    assert_size_stride(primals_374, (960, ), (1, ))
    assert_size_stride(primals_375, (40, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_376, (40, ), (1, ))
    assert_size_stride(primals_377, (960, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_378, (960, ), (1, ))
    assert_size_stride(primals_379, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_380, (160, ), (1, ))
    assert_size_stride(primals_381, (160, ), (1, ))
    assert_size_stride(primals_382, (160, ), (1, ))
    assert_size_stride(primals_383, (160, ), (1, ))
    assert_size_stride(primals_384, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_385, (960, ), (1, ))
    assert_size_stride(primals_386, (960, ), (1, ))
    assert_size_stride(primals_387, (960, ), (1, ))
    assert_size_stride(primals_388, (960, ), (1, ))
    assert_size_stride(primals_389, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_390, (960, ), (1, ))
    assert_size_stride(primals_391, (960, ), (1, ))
    assert_size_stride(primals_392, (960, ), (1, ))
    assert_size_stride(primals_393, (960, ), (1, ))
    assert_size_stride(primals_394, (40, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_395, (40, ), (1, ))
    assert_size_stride(primals_396, (960, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_397, (960, ), (1, ))
    assert_size_stride(primals_398, (256, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_399, (256, ), (1, ))
    assert_size_stride(primals_400, (256, ), (1, ))
    assert_size_stride(primals_401, (256, ), (1, ))
    assert_size_stride(primals_402, (256, ), (1, ))
    assert_size_stride(primals_403, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_404, (1536, ), (1, ))
    assert_size_stride(primals_405, (1536, ), (1, ))
    assert_size_stride(primals_406, (1536, ), (1, ))
    assert_size_stride(primals_407, (1536, ), (1, ))
    assert_size_stride(primals_408, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_409, (1536, ), (1, ))
    assert_size_stride(primals_410, (1536, ), (1, ))
    assert_size_stride(primals_411, (1536, ), (1, ))
    assert_size_stride(primals_412, (1536, ), (1, ))
    assert_size_stride(primals_413, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_414, (64, ), (1, ))
    assert_size_stride(primals_415, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_416, (1536, ), (1, ))
    assert_size_stride(primals_417, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_418, (256, ), (1, ))
    assert_size_stride(primals_419, (256, ), (1, ))
    assert_size_stride(primals_420, (256, ), (1, ))
    assert_size_stride(primals_421, (256, ), (1, ))
    assert_size_stride(primals_422, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_423, (1536, ), (1, ))
    assert_size_stride(primals_424, (1536, ), (1, ))
    assert_size_stride(primals_425, (1536, ), (1, ))
    assert_size_stride(primals_426, (1536, ), (1, ))
    assert_size_stride(primals_427, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_428, (1536, ), (1, ))
    assert_size_stride(primals_429, (1536, ), (1, ))
    assert_size_stride(primals_430, (1536, ), (1, ))
    assert_size_stride(primals_431, (1536, ), (1, ))
    assert_size_stride(primals_432, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_433, (64, ), (1, ))
    assert_size_stride(primals_434, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_435, (1536, ), (1, ))
    assert_size_stride(primals_436, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_437, (256, ), (1, ))
    assert_size_stride(primals_438, (256, ), (1, ))
    assert_size_stride(primals_439, (256, ), (1, ))
    assert_size_stride(primals_440, (256, ), (1, ))
    assert_size_stride(primals_441, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_442, (1536, ), (1, ))
    assert_size_stride(primals_443, (1536, ), (1, ))
    assert_size_stride(primals_444, (1536, ), (1, ))
    assert_size_stride(primals_445, (1536, ), (1, ))
    assert_size_stride(primals_446, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_447, (1536, ), (1, ))
    assert_size_stride(primals_448, (1536, ), (1, ))
    assert_size_stride(primals_449, (1536, ), (1, ))
    assert_size_stride(primals_450, (1536, ), (1, ))
    assert_size_stride(primals_451, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_452, (64, ), (1, ))
    assert_size_stride(primals_453, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_454, (1536, ), (1, ))
    assert_size_stride(primals_455, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_456, (256, ), (1, ))
    assert_size_stride(primals_457, (256, ), (1, ))
    assert_size_stride(primals_458, (256, ), (1, ))
    assert_size_stride(primals_459, (256, ), (1, ))
    assert_size_stride(primals_460, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_461, (1536, ), (1, ))
    assert_size_stride(primals_462, (1536, ), (1, ))
    assert_size_stride(primals_463, (1536, ), (1, ))
    assert_size_stride(primals_464, (1536, ), (1, ))
    assert_size_stride(primals_465, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_466, (1536, ), (1, ))
    assert_size_stride(primals_467, (1536, ), (1, ))
    assert_size_stride(primals_468, (1536, ), (1, ))
    assert_size_stride(primals_469, (1536, ), (1, ))
    assert_size_stride(primals_470, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_471, (64, ), (1, ))
    assert_size_stride(primals_472, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_473, (1536, ), (1, ))
    assert_size_stride(primals_474, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_475, (256, ), (1, ))
    assert_size_stride(primals_476, (256, ), (1, ))
    assert_size_stride(primals_477, (256, ), (1, ))
    assert_size_stride(primals_478, (256, ), (1, ))
    assert_size_stride(primals_479, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_480, (1536, ), (1, ))
    assert_size_stride(primals_481, (1536, ), (1, ))
    assert_size_stride(primals_482, (1536, ), (1, ))
    assert_size_stride(primals_483, (1536, ), (1, ))
    assert_size_stride(primals_484, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_485, (1536, ), (1, ))
    assert_size_stride(primals_486, (1536, ), (1, ))
    assert_size_stride(primals_487, (1536, ), (1, ))
    assert_size_stride(primals_488, (1536, ), (1, ))
    assert_size_stride(primals_489, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_490, (64, ), (1, ))
    assert_size_stride(primals_491, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_492, (1536, ), (1, ))
    assert_size_stride(primals_493, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_494, (256, ), (1, ))
    assert_size_stride(primals_495, (256, ), (1, ))
    assert_size_stride(primals_496, (256, ), (1, ))
    assert_size_stride(primals_497, (256, ), (1, ))
    assert_size_stride(primals_498, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_499, (1536, ), (1, ))
    assert_size_stride(primals_500, (1536, ), (1, ))
    assert_size_stride(primals_501, (1536, ), (1, ))
    assert_size_stride(primals_502, (1536, ), (1, ))
    assert_size_stride(primals_503, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_504, (1536, ), (1, ))
    assert_size_stride(primals_505, (1536, ), (1, ))
    assert_size_stride(primals_506, (1536, ), (1, ))
    assert_size_stride(primals_507, (1536, ), (1, ))
    assert_size_stride(primals_508, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_509, (64, ), (1, ))
    assert_size_stride(primals_510, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_511, (1536, ), (1, ))
    assert_size_stride(primals_512, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_513, (256, ), (1, ))
    assert_size_stride(primals_514, (256, ), (1, ))
    assert_size_stride(primals_515, (256, ), (1, ))
    assert_size_stride(primals_516, (256, ), (1, ))
    assert_size_stride(primals_517, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_518, (1536, ), (1, ))
    assert_size_stride(primals_519, (1536, ), (1, ))
    assert_size_stride(primals_520, (1536, ), (1, ))
    assert_size_stride(primals_521, (1536, ), (1, ))
    assert_size_stride(primals_522, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_523, (1536, ), (1, ))
    assert_size_stride(primals_524, (1536, ), (1, ))
    assert_size_stride(primals_525, (1536, ), (1, ))
    assert_size_stride(primals_526, (1536, ), (1, ))
    assert_size_stride(primals_527, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_528, (64, ), (1, ))
    assert_size_stride(primals_529, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_530, (1536, ), (1, ))
    assert_size_stride(primals_531, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_532, (256, ), (1, ))
    assert_size_stride(primals_533, (256, ), (1, ))
    assert_size_stride(primals_534, (256, ), (1, ))
    assert_size_stride(primals_535, (256, ), (1, ))
    assert_size_stride(primals_536, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_537, (1536, ), (1, ))
    assert_size_stride(primals_538, (1536, ), (1, ))
    assert_size_stride(primals_539, (1536, ), (1, ))
    assert_size_stride(primals_540, (1536, ), (1, ))
    assert_size_stride(primals_541, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_542, (1536, ), (1, ))
    assert_size_stride(primals_543, (1536, ), (1, ))
    assert_size_stride(primals_544, (1536, ), (1, ))
    assert_size_stride(primals_545, (1536, ), (1, ))
    assert_size_stride(primals_546, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_547, (64, ), (1, ))
    assert_size_stride(primals_548, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_549, (1536, ), (1, ))
    assert_size_stride(primals_550, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_551, (256, ), (1, ))
    assert_size_stride(primals_552, (256, ), (1, ))
    assert_size_stride(primals_553, (256, ), (1, ))
    assert_size_stride(primals_554, (256, ), (1, ))
    assert_size_stride(primals_555, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_556, (1536, ), (1, ))
    assert_size_stride(primals_557, (1536, ), (1, ))
    assert_size_stride(primals_558, (1536, ), (1, ))
    assert_size_stride(primals_559, (1536, ), (1, ))
    assert_size_stride(primals_560, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_561, (1536, ), (1, ))
    assert_size_stride(primals_562, (1536, ), (1, ))
    assert_size_stride(primals_563, (1536, ), (1, ))
    assert_size_stride(primals_564, (1536, ), (1, ))
    assert_size_stride(primals_565, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_566, (64, ), (1, ))
    assert_size_stride(primals_567, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_568, (1536, ), (1, ))
    assert_size_stride(primals_569, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_570, (256, ), (1, ))
    assert_size_stride(primals_571, (256, ), (1, ))
    assert_size_stride(primals_572, (256, ), (1, ))
    assert_size_stride(primals_573, (256, ), (1, ))
    assert_size_stride(primals_574, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_575, (1536, ), (1, ))
    assert_size_stride(primals_576, (1536, ), (1, ))
    assert_size_stride(primals_577, (1536, ), (1, ))
    assert_size_stride(primals_578, (1536, ), (1, ))
    assert_size_stride(primals_579, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_580, (1536, ), (1, ))
    assert_size_stride(primals_581, (1536, ), (1, ))
    assert_size_stride(primals_582, (1536, ), (1, ))
    assert_size_stride(primals_583, (1536, ), (1, ))
    assert_size_stride(primals_584, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_585, (64, ), (1, ))
    assert_size_stride(primals_586, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_587, (1536, ), (1, ))
    assert_size_stride(primals_588, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_589, (256, ), (1, ))
    assert_size_stride(primals_590, (256, ), (1, ))
    assert_size_stride(primals_591, (256, ), (1, ))
    assert_size_stride(primals_592, (256, ), (1, ))
    assert_size_stride(primals_593, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_594, (1536, ), (1, ))
    assert_size_stride(primals_595, (1536, ), (1, ))
    assert_size_stride(primals_596, (1536, ), (1, ))
    assert_size_stride(primals_597, (1536, ), (1, ))
    assert_size_stride(primals_598, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_599, (1536, ), (1, ))
    assert_size_stride(primals_600, (1536, ), (1, ))
    assert_size_stride(primals_601, (1536, ), (1, ))
    assert_size_stride(primals_602, (1536, ), (1, ))
    assert_size_stride(primals_603, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_604, (64, ), (1, ))
    assert_size_stride(primals_605, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_606, (1536, ), (1, ))
    assert_size_stride(primals_607, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_608, (256, ), (1, ))
    assert_size_stride(primals_609, (256, ), (1, ))
    assert_size_stride(primals_610, (256, ), (1, ))
    assert_size_stride(primals_611, (256, ), (1, ))
    assert_size_stride(primals_612, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_613, (1536, ), (1, ))
    assert_size_stride(primals_614, (1536, ), (1, ))
    assert_size_stride(primals_615, (1536, ), (1, ))
    assert_size_stride(primals_616, (1536, ), (1, ))
    assert_size_stride(primals_617, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_618, (1536, ), (1, ))
    assert_size_stride(primals_619, (1536, ), (1, ))
    assert_size_stride(primals_620, (1536, ), (1, ))
    assert_size_stride(primals_621, (1536, ), (1, ))
    assert_size_stride(primals_622, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_623, (64, ), (1, ))
    assert_size_stride(primals_624, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_625, (1536, ), (1, ))
    assert_size_stride(primals_626, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_627, (256, ), (1, ))
    assert_size_stride(primals_628, (256, ), (1, ))
    assert_size_stride(primals_629, (256, ), (1, ))
    assert_size_stride(primals_630, (256, ), (1, ))
    assert_size_stride(primals_631, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_632, (1536, ), (1, ))
    assert_size_stride(primals_633, (1536, ), (1, ))
    assert_size_stride(primals_634, (1536, ), (1, ))
    assert_size_stride(primals_635, (1536, ), (1, ))
    assert_size_stride(primals_636, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_637, (1536, ), (1, ))
    assert_size_stride(primals_638, (1536, ), (1, ))
    assert_size_stride(primals_639, (1536, ), (1, ))
    assert_size_stride(primals_640, (1536, ), (1, ))
    assert_size_stride(primals_641, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_642, (64, ), (1, ))
    assert_size_stride(primals_643, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_644, (1536, ), (1, ))
    assert_size_stride(primals_645, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_646, (256, ), (1, ))
    assert_size_stride(primals_647, (256, ), (1, ))
    assert_size_stride(primals_648, (256, ), (1, ))
    assert_size_stride(primals_649, (256, ), (1, ))
    assert_size_stride(primals_650, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_651, (1536, ), (1, ))
    assert_size_stride(primals_652, (1536, ), (1, ))
    assert_size_stride(primals_653, (1536, ), (1, ))
    assert_size_stride(primals_654, (1536, ), (1, ))
    assert_size_stride(primals_655, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_656, (1536, ), (1, ))
    assert_size_stride(primals_657, (1536, ), (1, ))
    assert_size_stride(primals_658, (1536, ), (1, ))
    assert_size_stride(primals_659, (1536, ), (1, ))
    assert_size_stride(primals_660, (64, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_661, (64, ), (1, ))
    assert_size_stride(primals_662, (1536, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_663, (1536, ), (1, ))
    assert_size_stride(primals_664, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_665, (256, ), (1, ))
    assert_size_stride(primals_666, (256, ), (1, ))
    assert_size_stride(primals_667, (256, ), (1, ))
    assert_size_stride(primals_668, (256, ), (1, ))
    assert_size_stride(primals_669, (1280, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_670, (1280, ), (1, ))
    assert_size_stride(primals_671, (1280, ), (1, ))
    assert_size_stride(primals_672, (1280, ), (1, ))
    assert_size_stride(primals_673, (1280, ), (1, ))
    assert_size_stride(primals_674, (16, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_675, (16, ), (1, ))
    assert_size_stride(primals_676, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((24, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_4, buf0, 72, 9, grid=grid(72, 9), stream=stream0)
        del primals_4
        buf1 = empty_strided_cuda((24, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_9, buf1, 576, 9, grid=grid(576, 9), stream=stream0)
        del primals_9
        buf2 = empty_strided_cuda((24, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_14, buf2, 576, 9, grid=grid(576, 9), stream=stream0)
        del primals_14
        buf3 = empty_strided_cuda((96, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_19, buf3, 2304, 9, grid=grid(2304, 9), stream=stream0)
        del primals_19
        buf4 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_29, buf4, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_29
        buf5 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_39, buf5, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_39
        buf6 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_49, buf6, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_49
        buf7 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_59, buf7, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_59
        buf8 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_69, buf8, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_69
        buf9 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_79, buf9, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_79
        buf10 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_89, buf10, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_89
        buf11 = empty_strided_cuda((4, 3, 4, 4), (48, 1, 12, 3), torch.float32)
        # Topologically Sorted Source Nodes: [mul, x, sub, x_1], Original ATen: [aten.mul, aten.add, aten.sub, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sub_5.run(primals_1, primals_2, primals_3, buf11, 12, 16, grid=grid(12, 16), stream=stream0)
        del primals_1
        del primals_2
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 24, 2, 2), (96, 1, 48, 24))
        buf13 = empty_strided_cuda((4, 24, 2, 2), (96, 1, 48, 24), torch.float32)
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_6.run(buf14, buf12, primals_5, primals_6, primals_7, primals_8, 384, grid=grid(384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 24, 2, 2), (96, 1, 48, 24))
        buf16 = empty_strided_cuda((4, 24, 2, 2), (96, 1, 48, 24), torch.float32)
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6, result], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7.run(buf17, buf15, primals_10, primals_11, primals_12, primals_13, buf14, 384, grid=grid(384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 24, 2, 2), (96, 1, 48, 24))
        buf19 = empty_strided_cuda((4, 24, 2, 2), (96, 1, 48, 24), torch.float32)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9, result_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7.run(buf20, buf18, primals_15, primals_16, primals_17, primals_18, buf17, 384, grid=grid(384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 96, 1, 1), (96, 1, 96, 96))
        buf22 = empty_strided_cuda((4, 96, 1, 1), (96, 1, 384, 384), torch.float32)
        buf23 = reinterpret_tensor(buf22, (4, 96, 1, 1), (96, 1, 96, 96), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_8.run(buf23, buf21, primals_20, primals_21, primals_22, primals_23, 384, grid=grid(384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_24, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 48, 1, 1), (48, 1, 48, 48))
        buf25 = empty_strided_cuda((4, 48, 1, 1), (48, 1, 48, 48), torch.float32)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf24, primals_25, primals_26, primals_27, primals_28, buf25, 192, grid=grid(192), stream=stream0)
        del primals_28
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 192, 1, 1), (192, 1, 192, 192))
        buf27 = empty_strided_cuda((4, 192, 1, 1), (192, 1, 768, 768), torch.float32)
        buf28 = reinterpret_tensor(buf27, (4, 192, 1, 1), (192, 1, 192, 192), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_10.run(buf28, buf26, primals_30, primals_31, primals_32, primals_33, 768, grid=grid(768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 48, 1, 1), (48, 1, 48, 48))
        buf30 = empty_strided_cuda((4, 48, 1, 1), (48, 1, 48, 48), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, result_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf29, primals_35, primals_36, primals_37, primals_38, buf25, buf30, 192, grid=grid(192), stream=stream0)
        del primals_38
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 192, 1, 1), (192, 1, 192, 192))
        buf32 = empty_strided_cuda((4, 192, 1, 1), (192, 1, 768, 768), torch.float32)
        buf33 = reinterpret_tensor(buf32, (4, 192, 1, 1), (192, 1, 192, 192), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_10.run(buf33, buf31, primals_40, primals_41, primals_42, primals_43, 768, grid=grid(768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 48, 1, 1), (48, 1, 48, 48))
        buf35 = empty_strided_cuda((4, 48, 1, 1), (48, 1, 48, 48), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, result_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf34, primals_45, primals_46, primals_47, primals_48, buf30, buf35, 192, grid=grid(192), stream=stream0)
        del primals_48
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 192, 1, 1), (192, 1, 192, 192))
        buf37 = empty_strided_cuda((4, 192, 1, 1), (192, 1, 768, 768), torch.float32)
        buf38 = reinterpret_tensor(buf37, (4, 192, 1, 1), (192, 1, 192, 192), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_10.run(buf38, buf36, primals_50, primals_51, primals_52, primals_53, 768, grid=grid(768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 48, 1, 1), (48, 1, 48, 48))
        buf40 = empty_strided_cuda((4, 48, 1, 1), (48, 1, 48, 48), torch.float32)
        # Topologically Sorted Source Nodes: [input_29, result_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf39, primals_55, primals_56, primals_57, primals_58, buf35, buf40, 192, grid=grid(192), stream=stream0)
        del primals_58
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 192, 1, 1), (192, 1, 192, 192))
        buf42 = empty_strided_cuda((4, 192, 1, 1), (192, 1, 768, 768), torch.float32)
        buf43 = reinterpret_tensor(buf42, (4, 192, 1, 1), (192, 1, 192, 192), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_10.run(buf43, buf41, primals_60, primals_61, primals_62, primals_63, 768, grid=grid(768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 64, 1, 1), (64, 1, 64, 64))
        buf45 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf44, primals_65, primals_66, primals_67, primals_68, buf45, 256, grid=grid(256), stream=stream0)
        del primals_68
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 256, 1, 1), (256, 1, 256, 256))
        buf47 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf48 = reinterpret_tensor(buf47, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf48, buf46, primals_70, primals_71, primals_72, primals_73, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 64, 1, 1), (64, 1, 64, 64))
        buf50 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_39, result_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf49, primals_75, primals_76, primals_77, primals_78, buf45, buf50, 256, grid=grid(256), stream=stream0)
        del primals_78
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 256, 1, 1), (256, 1, 256, 256))
        buf52 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf53 = reinterpret_tensor(buf52, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [input_41, input_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf53, buf51, primals_80, primals_81, primals_82, primals_83, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_84, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 64, 1, 1), (64, 1, 64, 64))
        buf55 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_44, result_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf54, primals_85, primals_86, primals_87, primals_88, buf50, buf55, 256, grid=grid(256), stream=stream0)
        del primals_88
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 256, 1, 1), (256, 1, 256, 256))
        buf57 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf58 = reinterpret_tensor(buf57, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [input_46, input_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf58, buf56, primals_90, primals_91, primals_92, primals_93, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 64, 1, 1), (64, 1, 64, 64))
        buf60 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_49, result_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf59, primals_95, primals_96, primals_97, primals_98, buf55, buf60, 256, grid=grid(256), stream=stream0)
        del primals_98
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_99, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 256, 1, 1), (256, 1, 256, 256))
        buf62 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf63 = reinterpret_tensor(buf62, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf63, buf61, primals_100, primals_101, primals_102, primals_103, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_104, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf64, (4, 256, 1, 1), (256, 1, 256, 256))
        buf65 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf66 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_54, input_55, scale], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_15.run(buf64, primals_105, primals_106, primals_107, primals_108, buf65, buf66, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_1], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 16, 1, 1), (16, 1, 16, 16))
        buf68 = buf67; del buf67  # reuse
        buf69 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [scale_1, scale_2], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_16.run(buf68, primals_110, buf69, 64, grid=grid(64), stream=stream0)
        del primals_110
        # Topologically Sorted Source Nodes: [scale_3], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 256, 1, 1), (256, 1, 256, 256))
        buf71 = buf70; del buf70  # reuse
        buf72 = reinterpret_tensor(buf65, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [input_55, scale_3, scale_4, input_56], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_17.run(buf71, buf72, primals_112, 1024, grid=grid(1024), stream=stream0)
        del primals_112
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 128, 1, 1), (128, 1, 128, 128))
        buf74 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf73, primals_114, primals_115, primals_116, primals_117, buf74, 512, grid=grid(512), stream=stream0)
        del primals_117
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 512, 1, 1), (512, 1, 512, 512))
        buf76 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf77 = reinterpret_tensor(buf76, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_19.run(buf77, buf75, primals_119, primals_120, primals_121, primals_122, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf78, (4, 512, 1, 1), (512, 1, 512, 512))
        buf79 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf80 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_63, input_64, scale_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_20.run(buf78, primals_124, primals_125, primals_126, primals_127, buf79, buf80, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_6], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 32, 1, 1), (32, 1, 32, 32))
        buf82 = buf81; del buf81  # reuse
        buf83 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [scale_6, scale_7], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(buf82, primals_129, buf83, 128, grid=grid(128), stream=stream0)
        del primals_129
        # Topologically Sorted Source Nodes: [scale_8], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 512, 1, 1), (512, 1, 512, 512))
        buf85 = buf84; del buf84  # reuse
        buf86 = reinterpret_tensor(buf79, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [input_64, scale_8, scale_9, input_65], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_22.run(buf85, buf86, primals_131, 2048, grid=grid(2048), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 128, 1, 1), (128, 1, 128, 128))
        buf88 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_67, result_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf87, primals_133, primals_134, primals_135, primals_136, buf74, buf88, 512, grid=grid(512), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 512, 1, 1), (512, 1, 512, 512))
        buf90 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf91 = reinterpret_tensor(buf90, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [input_69, input_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_19.run(buf91, buf89, primals_138, primals_139, primals_140, primals_141, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf92, (4, 512, 1, 1), (512, 1, 512, 512))
        buf93 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf94 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_72, input_73, scale_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_20.run(buf92, primals_143, primals_144, primals_145, primals_146, buf93, buf94, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_11], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 32, 1, 1), (32, 1, 32, 32))
        buf96 = buf95; del buf95  # reuse
        buf97 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [scale_11, scale_12], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(buf96, primals_148, buf97, 128, grid=grid(128), stream=stream0)
        del primals_148
        # Topologically Sorted Source Nodes: [scale_13], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 512, 1, 1), (512, 1, 512, 512))
        buf99 = buf98; del buf98  # reuse
        buf100 = reinterpret_tensor(buf93, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [input_73, scale_13, scale_14, input_74], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_22.run(buf99, buf100, primals_150, 2048, grid=grid(2048), stream=stream0)
        del primals_150
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 128, 1, 1), (128, 1, 128, 128))
        buf102 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_76, result_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf101, primals_152, primals_153, primals_154, primals_155, buf88, buf102, 512, grid=grid(512), stream=stream0)
        del primals_155
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 512, 1, 1), (512, 1, 512, 512))
        buf104 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf105 = reinterpret_tensor(buf104, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [input_78, input_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_19.run(buf105, buf103, primals_157, primals_158, primals_159, primals_160, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf106, (4, 512, 1, 1), (512, 1, 512, 512))
        buf107 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf108 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_81, input_82, scale_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_20.run(buf106, primals_162, primals_163, primals_164, primals_165, buf107, buf108, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_16], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 32, 1, 1), (32, 1, 32, 32))
        buf110 = buf109; del buf109  # reuse
        buf111 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [scale_16, scale_17], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(buf110, primals_167, buf111, 128, grid=grid(128), stream=stream0)
        del primals_167
        # Topologically Sorted Source Nodes: [scale_18], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 512, 1, 1), (512, 1, 512, 512))
        buf113 = buf112; del buf112  # reuse
        buf114 = reinterpret_tensor(buf107, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [input_82, scale_18, scale_19, input_83], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_22.run(buf113, buf114, primals_169, 2048, grid=grid(2048), stream=stream0)
        del primals_169
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 128, 1, 1), (128, 1, 128, 128))
        buf116 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_85, result_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf115, primals_171, primals_172, primals_173, primals_174, buf102, buf116, 512, grid=grid(512), stream=stream0)
        del primals_174
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 512, 1, 1), (512, 1, 512, 512))
        buf118 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf119 = reinterpret_tensor(buf118, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [input_87, input_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_19.run(buf119, buf117, primals_176, primals_177, primals_178, primals_179, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_180, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf120, (4, 512, 1, 1), (512, 1, 512, 512))
        buf121 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf122 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_90, input_91, scale_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_20.run(buf120, primals_181, primals_182, primals_183, primals_184, buf121, buf122, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_21], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 32, 1, 1), (32, 1, 32, 32))
        buf124 = buf123; del buf123  # reuse
        buf125 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [scale_21, scale_22], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(buf124, primals_186, buf125, 128, grid=grid(128), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [scale_23], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 512, 1, 1), (512, 1, 512, 512))
        buf127 = buf126; del buf126  # reuse
        buf128 = reinterpret_tensor(buf121, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [input_91, scale_23, scale_24, input_92], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_22.run(buf127, buf128, primals_188, 2048, grid=grid(2048), stream=stream0)
        del primals_188
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 128, 1, 1), (128, 1, 128, 128))
        buf130 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_94, result_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf129, primals_190, primals_191, primals_192, primals_193, buf116, buf130, 512, grid=grid(512), stream=stream0)
        del primals_193
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 512, 1, 1), (512, 1, 512, 512))
        buf132 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf133 = reinterpret_tensor(buf132, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [input_96, input_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_19.run(buf133, buf131, primals_195, primals_196, primals_197, primals_198, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf134, (4, 512, 1, 1), (512, 1, 512, 512))
        buf135 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf136 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_99, input_100, scale_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_20.run(buf134, primals_200, primals_201, primals_202, primals_203, buf135, buf136, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_26], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 32, 1, 1), (32, 1, 32, 32))
        buf138 = buf137; del buf137  # reuse
        buf139 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [scale_26, scale_27], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(buf138, primals_205, buf139, 128, grid=grid(128), stream=stream0)
        del primals_205
        # Topologically Sorted Source Nodes: [scale_28], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 512, 1, 1), (512, 1, 512, 512))
        buf141 = buf140; del buf140  # reuse
        buf142 = reinterpret_tensor(buf135, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [input_100, scale_28, scale_29, input_101], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_22.run(buf141, buf142, primals_207, 2048, grid=grid(2048), stream=stream0)
        del primals_207
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 128, 1, 1), (128, 1, 128, 128))
        buf144 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_103, result_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf143, primals_209, primals_210, primals_211, primals_212, buf130, buf144, 512, grid=grid(512), stream=stream0)
        del primals_212
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 768, 1, 1), (768, 1, 768, 768))
        buf146 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf147 = reinterpret_tensor(buf146, (4, 768, 1, 1), (768, 1, 768, 768), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [input_105, input_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_24.run(buf147, buf145, primals_214, primals_215, primals_216, primals_217, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_107], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_218, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf148, (4, 768, 1, 1), (768, 1, 768, 768))
        buf149 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf150 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 768, 768), torch.float32)
        # Topologically Sorted Source Nodes: [input_108, input_109, scale_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_25.run(buf148, primals_219, primals_220, primals_221, primals_222, buf149, buf150, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_31], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 32, 1, 1), (32, 1, 32, 32))
        buf152 = buf151; del buf151  # reuse
        buf153 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [scale_31, scale_32], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(buf152, primals_224, buf153, 128, grid=grid(128), stream=stream0)
        del primals_224
        # Topologically Sorted Source Nodes: [scale_33], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 768, 1, 1), (768, 1, 768, 768))
        buf155 = buf154; del buf154  # reuse
        buf156 = reinterpret_tensor(buf149, (4, 768, 1, 1), (768, 1, 768, 768), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [input_109, scale_33, scale_34, input_110], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_26.run(buf155, buf156, primals_226, 3072, grid=grid(3072), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [input_111], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 160, 1, 1), (160, 1, 160, 160))
        buf158 = empty_strided_cuda((4, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf157, primals_228, primals_229, primals_230, primals_231, buf158, 640, grid=grid(640), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [input_113], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 960, 1, 1), (960, 1, 960, 960))
        buf160 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf161 = reinterpret_tensor(buf160, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [input_114, input_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf161, buf159, primals_233, primals_234, primals_235, primals_236, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf162, (4, 960, 1, 1), (960, 1, 960, 960))
        buf163 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf164 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_117, input_118, scale_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf162, primals_238, primals_239, primals_240, primals_241, buf163, buf164, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_36], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 40, 1, 1), (40, 1, 40, 40))
        buf166 = buf165; del buf165  # reuse
        buf167 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [scale_36, scale_37], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_30.run(buf166, primals_243, buf167, 160, grid=grid(160), stream=stream0)
        del primals_243
        # Topologically Sorted Source Nodes: [scale_38], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_244, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 960, 1, 1), (960, 1, 960, 960))
        buf169 = buf168; del buf168  # reuse
        buf170 = reinterpret_tensor(buf163, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [input_118, scale_38, scale_39, input_119], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_31.run(buf169, buf170, primals_245, 3840, grid=grid(3840), stream=stream0)
        del primals_245
        # Topologically Sorted Source Nodes: [input_120], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 160, 1, 1), (160, 1, 160, 160))
        buf172 = empty_strided_cuda((4, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_121, result_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf171, primals_247, primals_248, primals_249, primals_250, buf158, buf172, 640, grid=grid(640), stream=stream0)
        del primals_250
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 960, 1, 1), (960, 1, 960, 960))
        buf174 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf175 = reinterpret_tensor(buf174, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [input_123, input_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf175, buf173, primals_252, primals_253, primals_254, primals_255, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf176, (4, 960, 1, 1), (960, 1, 960, 960))
        buf177 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf178 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_126, input_127, scale_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf176, primals_257, primals_258, primals_259, primals_260, buf177, buf178, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_41], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 40, 1, 1), (40, 1, 40, 40))
        buf180 = buf179; del buf179  # reuse
        buf181 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [scale_41, scale_42], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_30.run(buf180, primals_262, buf181, 160, grid=grid(160), stream=stream0)
        del primals_262
        # Topologically Sorted Source Nodes: [scale_43], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_263, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 960, 1, 1), (960, 1, 960, 960))
        buf183 = buf182; del buf182  # reuse
        buf184 = reinterpret_tensor(buf177, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [input_127, scale_43, scale_44, input_128], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_31.run(buf183, buf184, primals_264, 3840, grid=grid(3840), stream=stream0)
        del primals_264
        # Topologically Sorted Source Nodes: [input_129], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 160, 1, 1), (160, 1, 160, 160))
        buf186 = empty_strided_cuda((4, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_130, result_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf185, primals_266, primals_267, primals_268, primals_269, buf172, buf186, 640, grid=grid(640), stream=stream0)
        del primals_269
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 960, 1, 1), (960, 1, 960, 960))
        buf188 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf189 = reinterpret_tensor(buf188, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [input_132, input_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf189, buf187, primals_271, primals_272, primals_273, primals_274, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_275, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf190, (4, 960, 1, 1), (960, 1, 960, 960))
        buf191 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf192 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_135, input_136, scale_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf190, primals_276, primals_277, primals_278, primals_279, buf191, buf192, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_46], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 40, 1, 1), (40, 1, 40, 40))
        buf194 = buf193; del buf193  # reuse
        buf195 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [scale_46, scale_47], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_30.run(buf194, primals_281, buf195, 160, grid=grid(160), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [scale_48], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 960, 1, 1), (960, 1, 960, 960))
        buf197 = buf196; del buf196  # reuse
        buf198 = reinterpret_tensor(buf191, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [input_136, scale_48, scale_49, input_137], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_31.run(buf197, buf198, primals_283, 3840, grid=grid(3840), stream=stream0)
        del primals_283
        # Topologically Sorted Source Nodes: [input_138], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_284, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 160, 1, 1), (160, 1, 160, 160))
        buf200 = empty_strided_cuda((4, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_139, result_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf199, primals_285, primals_286, primals_287, primals_288, buf186, buf200, 640, grid=grid(640), stream=stream0)
        del primals_288
        # Topologically Sorted Source Nodes: [input_140], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_289, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 960, 1, 1), (960, 1, 960, 960))
        buf202 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf203 = reinterpret_tensor(buf202, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [input_141, input_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf203, buf201, primals_290, primals_291, primals_292, primals_293, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [input_143], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf204, (4, 960, 1, 1), (960, 1, 960, 960))
        buf205 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf206 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_144, input_145, scale_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf204, primals_295, primals_296, primals_297, primals_298, buf205, buf206, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_51], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_299, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 40, 1, 1), (40, 1, 40, 40))
        buf208 = buf207; del buf207  # reuse
        buf209 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [scale_51, scale_52], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_30.run(buf208, primals_300, buf209, 160, grid=grid(160), stream=stream0)
        del primals_300
        # Topologically Sorted Source Nodes: [scale_53], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 960, 1, 1), (960, 1, 960, 960))
        buf211 = buf210; del buf210  # reuse
        buf212 = reinterpret_tensor(buf205, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [input_145, scale_53, scale_54, input_146], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_31.run(buf211, buf212, primals_302, 3840, grid=grid(3840), stream=stream0)
        del primals_302
        # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 160, 1, 1), (160, 1, 160, 160))
        buf214 = empty_strided_cuda((4, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_148, result_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf213, primals_304, primals_305, primals_306, primals_307, buf200, buf214, 640, grid=grid(640), stream=stream0)
        del primals_307
        # Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_308, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 960, 1, 1), (960, 1, 960, 960))
        buf216 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf217 = reinterpret_tensor(buf216, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [input_150, input_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf217, buf215, primals_309, primals_310, primals_311, primals_312, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [input_152], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_313, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf218, (4, 960, 1, 1), (960, 1, 960, 960))
        buf219 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf220 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_153, input_154, scale_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf218, primals_314, primals_315, primals_316, primals_317, buf219, buf220, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_56], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_318, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 40, 1, 1), (40, 1, 40, 40))
        buf222 = buf221; del buf221  # reuse
        buf223 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [scale_56, scale_57], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_30.run(buf222, primals_319, buf223, 160, grid=grid(160), stream=stream0)
        del primals_319
        # Topologically Sorted Source Nodes: [scale_58], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_320, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 960, 1, 1), (960, 1, 960, 960))
        buf225 = buf224; del buf224  # reuse
        buf226 = reinterpret_tensor(buf219, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [input_154, scale_58, scale_59, input_155], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_31.run(buf225, buf226, primals_321, 3840, grid=grid(3840), stream=stream0)
        del primals_321
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 160, 1, 1), (160, 1, 160, 160))
        buf228 = empty_strided_cuda((4, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_157, result_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf227, primals_323, primals_324, primals_325, primals_326, buf214, buf228, 640, grid=grid(640), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [input_158], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 960, 1, 1), (960, 1, 960, 960))
        buf230 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf231 = reinterpret_tensor(buf230, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [input_159, input_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf231, buf229, primals_328, primals_329, primals_330, primals_331, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [input_161], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf232, (4, 960, 1, 1), (960, 1, 960, 960))
        buf233 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf234 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_162, input_163, scale_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf232, primals_333, primals_334, primals_335, primals_336, buf233, buf234, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_61], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_337, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 40, 1, 1), (40, 1, 40, 40))
        buf236 = buf235; del buf235  # reuse
        buf237 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [scale_61, scale_62], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_30.run(buf236, primals_338, buf237, 160, grid=grid(160), stream=stream0)
        del primals_338
        # Topologically Sorted Source Nodes: [scale_63], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_339, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 960, 1, 1), (960, 1, 960, 960))
        buf239 = buf238; del buf238  # reuse
        buf240 = reinterpret_tensor(buf233, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [input_163, scale_63, scale_64, input_164], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_31.run(buf239, buf240, primals_340, 3840, grid=grid(3840), stream=stream0)
        del primals_340
        # Topologically Sorted Source Nodes: [input_165], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_341, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 160, 1, 1), (160, 1, 160, 160))
        buf242 = empty_strided_cuda((4, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_166, result_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf241, primals_342, primals_343, primals_344, primals_345, buf228, buf242, 640, grid=grid(640), stream=stream0)
        del primals_345
        # Topologically Sorted Source Nodes: [input_167], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_346, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 960, 1, 1), (960, 1, 960, 960))
        buf244 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf245 = reinterpret_tensor(buf244, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [input_168, input_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf245, buf243, primals_347, primals_348, primals_349, primals_350, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [input_170], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, primals_351, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf246, (4, 960, 1, 1), (960, 1, 960, 960))
        buf247 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf248 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_171, input_172, scale_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf246, primals_352, primals_353, primals_354, primals_355, buf247, buf248, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_66], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, primals_356, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (4, 40, 1, 1), (40, 1, 40, 40))
        buf250 = buf249; del buf249  # reuse
        buf251 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [scale_66, scale_67], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_30.run(buf250, primals_357, buf251, 160, grid=grid(160), stream=stream0)
        del primals_357
        # Topologically Sorted Source Nodes: [scale_68], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_358, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 960, 1, 1), (960, 1, 960, 960))
        buf253 = buf252; del buf252  # reuse
        buf254 = reinterpret_tensor(buf247, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [input_172, scale_68, scale_69, input_173], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_31.run(buf253, buf254, primals_359, 3840, grid=grid(3840), stream=stream0)
        del primals_359
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_360, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 160, 1, 1), (160, 1, 160, 160))
        buf256 = empty_strided_cuda((4, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_175, result_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf255, primals_361, primals_362, primals_363, primals_364, buf242, buf256, 640, grid=grid(640), stream=stream0)
        del primals_364
        # Topologically Sorted Source Nodes: [input_176], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_365, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 960, 1, 1), (960, 1, 960, 960))
        buf258 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf259 = reinterpret_tensor(buf258, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [input_177, input_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf259, buf257, primals_366, primals_367, primals_368, primals_369, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [input_179], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_370, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf260, (4, 960, 1, 1), (960, 1, 960, 960))
        buf261 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf262 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_180, input_181, scale_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf260, primals_371, primals_372, primals_373, primals_374, buf261, buf262, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_71], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_375, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 40, 1, 1), (40, 1, 40, 40))
        buf264 = buf263; del buf263  # reuse
        buf265 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [scale_71, scale_72], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_30.run(buf264, primals_376, buf265, 160, grid=grid(160), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [scale_73], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_377, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 960, 1, 1), (960, 1, 960, 960))
        buf267 = buf266; del buf266  # reuse
        buf268 = reinterpret_tensor(buf261, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [input_181, scale_73, scale_74, input_182], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_31.run(buf267, buf268, primals_378, 3840, grid=grid(3840), stream=stream0)
        del primals_378
        # Topologically Sorted Source Nodes: [input_183], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_379, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (4, 160, 1, 1), (160, 1, 160, 160))
        buf270 = empty_strided_cuda((4, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_184, result_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf269, primals_380, primals_381, primals_382, primals_383, buf256, buf270, 640, grid=grid(640), stream=stream0)
        del primals_383
        # Topologically Sorted Source Nodes: [input_185], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_384, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 960, 1, 1), (960, 1, 960, 960))
        buf272 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf273 = reinterpret_tensor(buf272, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [input_186, input_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf273, buf271, primals_385, primals_386, primals_387, primals_388, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [input_188], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_389, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf274, (4, 960, 1, 1), (960, 1, 960, 960))
        buf275 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf276 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_189, input_190, scale_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf274, primals_390, primals_391, primals_392, primals_393, buf275, buf276, 3840, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_76], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_394, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 40, 1, 1), (40, 1, 40, 40))
        buf278 = buf277; del buf277  # reuse
        buf279 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [scale_76, scale_77], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_30.run(buf278, primals_395, buf279, 160, grid=grid(160), stream=stream0)
        del primals_395
        # Topologically Sorted Source Nodes: [scale_78], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_396, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 960, 1, 1), (960, 1, 960, 960))
        buf281 = buf280; del buf280  # reuse
        buf282 = reinterpret_tensor(buf275, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [input_190, scale_78, scale_79, input_191], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_31.run(buf281, buf282, primals_397, 3840, grid=grid(3840), stream=stream0)
        del primals_397
        # Topologically Sorted Source Nodes: [input_192], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_398, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 256, 1, 1), (256, 1, 256, 256))
        buf284 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_193], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf283, primals_399, primals_400, primals_401, primals_402, buf284, 1024, grid=grid(1024), stream=stream0)
        del primals_402
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_403, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf286 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf287 = reinterpret_tensor(buf286, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [input_195, input_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf287, buf285, primals_404, primals_405, primals_406, primals_407, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_197], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_408, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf288, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf289 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf290 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_198, input_199, scale_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf288, primals_409, primals_410, primals_411, primals_412, buf289, buf290, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_81], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_413, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 64, 1, 1), (64, 1, 64, 64))
        buf292 = buf291; del buf291  # reuse
        buf293 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_81, scale_82], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf292, primals_414, buf293, 256, grid=grid(256), stream=stream0)
        del primals_414
        # Topologically Sorted Source Nodes: [scale_83], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_415, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf295 = buf294; del buf294  # reuse
        buf296 = reinterpret_tensor(buf289, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [input_199, scale_83, scale_84, input_200], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf295, buf296, primals_416, 6144, grid=grid(6144), stream=stream0)
        del primals_416
        # Topologically Sorted Source Nodes: [input_201], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, primals_417, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (4, 256, 1, 1), (256, 1, 256, 256))
        buf298 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_202, result_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf297, primals_418, primals_419, primals_420, primals_421, buf284, buf298, 1024, grid=grid(1024), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [input_203], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf300 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf301 = reinterpret_tensor(buf300, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [input_204, input_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf301, buf299, primals_423, primals_424, primals_425, primals_426, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_206], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf302, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf303 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf304 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_207, input_208, scale_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf302, primals_428, primals_429, primals_430, primals_431, buf303, buf304, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_86], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, primals_432, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (4, 64, 1, 1), (64, 1, 64, 64))
        buf306 = buf305; del buf305  # reuse
        buf307 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_86, scale_87], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf306, primals_433, buf307, 256, grid=grid(256), stream=stream0)
        del primals_433
        # Topologically Sorted Source Nodes: [scale_88], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_434, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf309 = buf308; del buf308  # reuse
        buf310 = reinterpret_tensor(buf303, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [input_208, scale_88, scale_89, input_209], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf309, buf310, primals_435, 6144, grid=grid(6144), stream=stream0)
        del primals_435
        # Topologically Sorted Source Nodes: [input_210], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, primals_436, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (4, 256, 1, 1), (256, 1, 256, 256))
        buf312 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_211, result_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf311, primals_437, primals_438, primals_439, primals_440, buf298, buf312, 1024, grid=grid(1024), stream=stream0)
        del primals_440
        # Topologically Sorted Source Nodes: [input_212], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, primals_441, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf314 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf315 = reinterpret_tensor(buf314, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [input_213, input_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf315, buf313, primals_442, primals_443, primals_444, primals_445, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_215], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, primals_446, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf316, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf317 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf318 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_216, input_217, scale_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf316, primals_447, primals_448, primals_449, primals_450, buf317, buf318, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_91], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_451, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 64, 1, 1), (64, 1, 64, 64))
        buf320 = buf319; del buf319  # reuse
        buf321 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_91, scale_92], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf320, primals_452, buf321, 256, grid=grid(256), stream=stream0)
        del primals_452
        # Topologically Sorted Source Nodes: [scale_93], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_453, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf323 = buf322; del buf322  # reuse
        buf324 = reinterpret_tensor(buf317, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [input_217, scale_93, scale_94, input_218], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf323, buf324, primals_454, 6144, grid=grid(6144), stream=stream0)
        del primals_454
        # Topologically Sorted Source Nodes: [input_219], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf324, primals_455, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (4, 256, 1, 1), (256, 1, 256, 256))
        buf326 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_220, result_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf325, primals_456, primals_457, primals_458, primals_459, buf312, buf326, 1024, grid=grid(1024), stream=stream0)
        del primals_459
        # Topologically Sorted Source Nodes: [input_221], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, primals_460, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf328 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf329 = reinterpret_tensor(buf328, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [input_222, input_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf329, buf327, primals_461, primals_462, primals_463, primals_464, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_224], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_465, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf330, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf331 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf332 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_225, input_226, scale_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf330, primals_466, primals_467, primals_468, primals_469, buf331, buf332, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_96], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_470, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 64, 1, 1), (64, 1, 64, 64))
        buf334 = buf333; del buf333  # reuse
        buf335 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_96, scale_97], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf334, primals_471, buf335, 256, grid=grid(256), stream=stream0)
        del primals_471
        # Topologically Sorted Source Nodes: [scale_98], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf337 = buf336; del buf336  # reuse
        buf338 = reinterpret_tensor(buf331, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [input_226, scale_98, scale_99, input_227], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf337, buf338, primals_473, 6144, grid=grid(6144), stream=stream0)
        del primals_473
        # Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_474, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (4, 256, 1, 1), (256, 1, 256, 256))
        buf340 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_229, result_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf339, primals_475, primals_476, primals_477, primals_478, buf326, buf340, 1024, grid=grid(1024), stream=stream0)
        del primals_478
        # Topologically Sorted Source Nodes: [input_230], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf340, primals_479, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf342 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf343 = reinterpret_tensor(buf342, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [input_231, input_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf343, buf341, primals_480, primals_481, primals_482, primals_483, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_233], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_484, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf344, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf345 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf346 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_234, input_235, scale_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf344, primals_485, primals_486, primals_487, primals_488, buf345, buf346, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_101], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_489, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (4, 64, 1, 1), (64, 1, 64, 64))
        buf348 = buf347; del buf347  # reuse
        buf349 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_101, scale_102], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf348, primals_490, buf349, 256, grid=grid(256), stream=stream0)
        del primals_490
        # Topologically Sorted Source Nodes: [scale_103], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_491, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf351 = buf350; del buf350  # reuse
        buf352 = reinterpret_tensor(buf345, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [input_235, scale_103, scale_104, input_236], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf351, buf352, primals_492, 6144, grid=grid(6144), stream=stream0)
        del primals_492
        # Topologically Sorted Source Nodes: [input_237], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_493, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (4, 256, 1, 1), (256, 1, 256, 256))
        buf354 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_238, result_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf353, primals_494, primals_495, primals_496, primals_497, buf340, buf354, 1024, grid=grid(1024), stream=stream0)
        del primals_497
        # Topologically Sorted Source Nodes: [input_239], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, primals_498, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf356 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf357 = reinterpret_tensor(buf356, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [input_240, input_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf357, buf355, primals_499, primals_500, primals_501, primals_502, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_242], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, primals_503, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf358, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf359 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf360 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_243, input_244, scale_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf358, primals_504, primals_505, primals_506, primals_507, buf359, buf360, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_106], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, primals_508, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (4, 64, 1, 1), (64, 1, 64, 64))
        buf362 = buf361; del buf361  # reuse
        buf363 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_106, scale_107], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf362, primals_509, buf363, 256, grid=grid(256), stream=stream0)
        del primals_509
        # Topologically Sorted Source Nodes: [scale_108], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, primals_510, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf365 = buf364; del buf364  # reuse
        buf366 = reinterpret_tensor(buf359, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [input_244, scale_108, scale_109, input_245], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf365, buf366, primals_511, 6144, grid=grid(6144), stream=stream0)
        del primals_511
        # Topologically Sorted Source Nodes: [input_246], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (4, 256, 1, 1), (256, 1, 256, 256))
        buf368 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_247, result_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf367, primals_513, primals_514, primals_515, primals_516, buf354, buf368, 1024, grid=grid(1024), stream=stream0)
        del primals_516
        # Topologically Sorted Source Nodes: [input_248], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, primals_517, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf370 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf371 = reinterpret_tensor(buf370, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [input_249, input_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf371, buf369, primals_518, primals_519, primals_520, primals_521, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_522, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf372, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf373 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf374 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_252, input_253, scale_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf372, primals_523, primals_524, primals_525, primals_526, buf373, buf374, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_111], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf374, primals_527, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (4, 64, 1, 1), (64, 1, 64, 64))
        buf376 = buf375; del buf375  # reuse
        buf377 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_111, scale_112], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf376, primals_528, buf377, 256, grid=grid(256), stream=stream0)
        del primals_528
        # Topologically Sorted Source Nodes: [scale_113], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_529, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf379 = buf378; del buf378  # reuse
        buf380 = reinterpret_tensor(buf373, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [input_253, scale_113, scale_114, input_254], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf379, buf380, primals_530, 6144, grid=grid(6144), stream=stream0)
        del primals_530
        # Topologically Sorted Source Nodes: [input_255], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, primals_531, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 256, 1, 1), (256, 1, 256, 256))
        buf382 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_256, result_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf381, primals_532, primals_533, primals_534, primals_535, buf368, buf382, 1024, grid=grid(1024), stream=stream0)
        del primals_535
        # Topologically Sorted Source Nodes: [input_257], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_536, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf384 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf385 = reinterpret_tensor(buf384, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [input_258, input_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf385, buf383, primals_537, primals_538, primals_539, primals_540, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_260], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, primals_541, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf386, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf387 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf388 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_261, input_262, scale_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf386, primals_542, primals_543, primals_544, primals_545, buf387, buf388, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_116], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_546, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (4, 64, 1, 1), (64, 1, 64, 64))
        buf390 = buf389; del buf389  # reuse
        buf391 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_116, scale_117], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf390, primals_547, buf391, 256, grid=grid(256), stream=stream0)
        del primals_547
        # Topologically Sorted Source Nodes: [scale_118], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_548, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf393 = buf392; del buf392  # reuse
        buf394 = reinterpret_tensor(buf387, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [input_262, scale_118, scale_119, input_263], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf393, buf394, primals_549, 6144, grid=grid(6144), stream=stream0)
        del primals_549
        # Topologically Sorted Source Nodes: [input_264], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_550, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (4, 256, 1, 1), (256, 1, 256, 256))
        buf396 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_265, result_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf395, primals_551, primals_552, primals_553, primals_554, buf382, buf396, 1024, grid=grid(1024), stream=stream0)
        del primals_554
        # Topologically Sorted Source Nodes: [input_266], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, primals_555, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf398 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf399 = reinterpret_tensor(buf398, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [input_267, input_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf399, buf397, primals_556, primals_557, primals_558, primals_559, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_269], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf399, primals_560, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf400, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf401 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf402 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_270, input_271, scale_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf400, primals_561, primals_562, primals_563, primals_564, buf401, buf402, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_121], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, primals_565, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (4, 64, 1, 1), (64, 1, 64, 64))
        buf404 = buf403; del buf403  # reuse
        buf405 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_121, scale_122], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf404, primals_566, buf405, 256, grid=grid(256), stream=stream0)
        del primals_566
        # Topologically Sorted Source Nodes: [scale_123], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf405, primals_567, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf407 = buf406; del buf406  # reuse
        buf408 = reinterpret_tensor(buf401, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [input_271, scale_123, scale_124, input_272], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf407, buf408, primals_568, 6144, grid=grid(6144), stream=stream0)
        del primals_568
        # Topologically Sorted Source Nodes: [input_273], Original ATen: [aten.convolution]
        buf409 = extern_kernels.convolution(buf408, primals_569, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf409, (4, 256, 1, 1), (256, 1, 256, 256))
        buf410 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_274, result_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf409, primals_570, primals_571, primals_572, primals_573, buf396, buf410, 1024, grid=grid(1024), stream=stream0)
        del primals_573
        # Topologically Sorted Source Nodes: [input_275], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, primals_574, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf412 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf413 = reinterpret_tensor(buf412, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [input_276, input_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf413, buf411, primals_575, primals_576, primals_577, primals_578, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_278], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_579, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf414, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf415 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf416 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_279, input_280, scale_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf414, primals_580, primals_581, primals_582, primals_583, buf415, buf416, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_126], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf416, primals_584, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (4, 64, 1, 1), (64, 1, 64, 64))
        buf418 = buf417; del buf417  # reuse
        buf419 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_126, scale_127], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf418, primals_585, buf419, 256, grid=grid(256), stream=stream0)
        del primals_585
        # Topologically Sorted Source Nodes: [scale_128], Original ATen: [aten.convolution]
        buf420 = extern_kernels.convolution(buf419, primals_586, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf421 = buf420; del buf420  # reuse
        buf422 = reinterpret_tensor(buf415, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [input_280, scale_128, scale_129, input_281], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf421, buf422, primals_587, 6144, grid=grid(6144), stream=stream0)
        del primals_587
        # Topologically Sorted Source Nodes: [input_282], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf422, primals_588, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf423, (4, 256, 1, 1), (256, 1, 256, 256))
        buf424 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_283, result_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf423, primals_589, primals_590, primals_591, primals_592, buf410, buf424, 1024, grid=grid(1024), stream=stream0)
        del primals_592
        # Topologically Sorted Source Nodes: [input_284], Original ATen: [aten.convolution]
        buf425 = extern_kernels.convolution(buf424, primals_593, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf425, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf426 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf427 = reinterpret_tensor(buf426, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [input_285, input_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf427, buf425, primals_594, primals_595, primals_596, primals_597, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, primals_598, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf428, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf429 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf430 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_288, input_289, scale_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf428, primals_599, primals_600, primals_601, primals_602, buf429, buf430, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_131], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_603, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (4, 64, 1, 1), (64, 1, 64, 64))
        buf432 = buf431; del buf431  # reuse
        buf433 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_131, scale_132], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf432, primals_604, buf433, 256, grid=grid(256), stream=stream0)
        del primals_604
        # Topologically Sorted Source Nodes: [scale_133], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, primals_605, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf435 = buf434; del buf434  # reuse
        buf436 = reinterpret_tensor(buf429, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [input_289, scale_133, scale_134, input_290], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf435, buf436, primals_606, 6144, grid=grid(6144), stream=stream0)
        del primals_606
        # Topologically Sorted Source Nodes: [input_291], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, primals_607, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (4, 256, 1, 1), (256, 1, 256, 256))
        buf438 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_292, result_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf437, primals_608, primals_609, primals_610, primals_611, buf424, buf438, 1024, grid=grid(1024), stream=stream0)
        del primals_611
        # Topologically Sorted Source Nodes: [input_293], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, primals_612, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf440 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf441 = reinterpret_tensor(buf440, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [input_294, input_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf441, buf439, primals_613, primals_614, primals_615, primals_616, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_296], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf441, primals_617, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf442, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf443 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf444 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_297, input_298, scale_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf442, primals_618, primals_619, primals_620, primals_621, buf443, buf444, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_136], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, primals_622, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (4, 64, 1, 1), (64, 1, 64, 64))
        buf446 = buf445; del buf445  # reuse
        buf447 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_136, scale_137], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf446, primals_623, buf447, 256, grid=grid(256), stream=stream0)
        del primals_623
        # Topologically Sorted Source Nodes: [scale_138], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, primals_624, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf449 = buf448; del buf448  # reuse
        buf450 = reinterpret_tensor(buf443, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [input_298, scale_138, scale_139, input_299], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf449, buf450, primals_625, 6144, grid=grid(6144), stream=stream0)
        del primals_625
        # Topologically Sorted Source Nodes: [input_300], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, primals_626, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (4, 256, 1, 1), (256, 1, 256, 256))
        buf452 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_301, result_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf451, primals_627, primals_628, primals_629, primals_630, buf438, buf452, 1024, grid=grid(1024), stream=stream0)
        del primals_630
        # Topologically Sorted Source Nodes: [input_302], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_631, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf454 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf455 = reinterpret_tensor(buf454, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [input_303, input_304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf455, buf453, primals_632, primals_633, primals_634, primals_635, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_305], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_636, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf456, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf457 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf458 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_306, input_307, scale_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf456, primals_637, primals_638, primals_639, primals_640, buf457, buf458, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_141], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf458, primals_641, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (4, 64, 1, 1), (64, 1, 64, 64))
        buf460 = buf459; del buf459  # reuse
        buf461 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_141, scale_142], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf460, primals_642, buf461, 256, grid=grid(256), stream=stream0)
        del primals_642
        # Topologically Sorted Source Nodes: [scale_143], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_643, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf463 = buf462; del buf462  # reuse
        buf464 = reinterpret_tensor(buf457, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [input_307, scale_143, scale_144, input_308], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf463, buf464, primals_644, 6144, grid=grid(6144), stream=stream0)
        del primals_644
        # Topologically Sorted Source Nodes: [input_309], Original ATen: [aten.convolution]
        buf465 = extern_kernels.convolution(buf464, primals_645, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (4, 256, 1, 1), (256, 1, 256, 256))
        buf466 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_310, result_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf465, primals_646, primals_647, primals_648, primals_649, buf452, buf466, 1024, grid=grid(1024), stream=stream0)
        del primals_649
        # Topologically Sorted Source Nodes: [input_311], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, primals_650, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf468 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf469 = reinterpret_tensor(buf468, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [input_312, input_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf469, buf467, primals_651, primals_652, primals_653, primals_654, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_314], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, primals_655, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf470, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf471 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf472 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_315, input_316, scale_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf470, primals_656, primals_657, primals_658, primals_659, buf471, buf472, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_146], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, primals_660, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (4, 64, 1, 1), (64, 1, 64, 64))
        buf474 = buf473; del buf473  # reuse
        buf475 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scale_146, scale_147], Original ATen: [aten.convolution, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_36.run(buf474, primals_661, buf475, 256, grid=grid(256), stream=stream0)
        del primals_661
        # Topologically Sorted Source Nodes: [scale_148], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_662, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf477 = buf476; del buf476  # reuse
        buf478 = reinterpret_tensor(buf471, (4, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [input_316, scale_148, scale_149, input_317], Original ATen: [aten.silu, aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_silu_37.run(buf477, buf478, primals_663, 6144, grid=grid(6144), stream=stream0)
        del primals_663
        # Topologically Sorted Source Nodes: [input_318], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, primals_664, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (4, 256, 1, 1), (256, 1, 256, 256))
        buf480 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_319, result_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf479, primals_665, primals_666, primals_667, primals_668, buf466, buf480, 1024, grid=grid(1024), stream=stream0)
        del primals_668
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, primals_669, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (4, 1280, 1, 1), (1280, 1, 1280, 1280))
        buf482 = empty_strided_cuda((4, 1280, 1, 1), (1280, 1, 5120, 5120), torch.float32)
        buf483 = reinterpret_tensor(buf482, (4, 1280, 1, 1), (1280, 1, 1280, 1280), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [input_321, input_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_39.run(buf483, buf481, primals_670, primals_671, primals_672, primals_673, 5120, grid=grid(5120), stream=stream0)
        # Topologically Sorted Source Nodes: [input_323], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, primals_674, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (4, 16, 1, 1), (16, 1, 16, 16))
        buf485 = reinterpret_tensor(buf484, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [input_324], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_40.run(buf485, primals_675, primals_676, 64, grid=grid(64), stream=stream0)
        del primals_675
    return (buf485, primals_3, buf0, primals_5, primals_6, primals_7, primals_8, buf1, primals_10, primals_11, primals_12, primals_13, buf2, primals_15, primals_16, primals_17, primals_18, buf3, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, buf4, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, buf5, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, buf6, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, buf7, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, buf8, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, buf9, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, buf10, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_111, primals_113, primals_114, primals_115, primals_116, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_206, primals_208, primals_209, primals_210, primals_211, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_244, primals_246, primals_247, primals_248, primals_249, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_263, primals_265, primals_266, primals_267, primals_268, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_282, primals_284, primals_285, primals_286, primals_287, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_301, primals_303, primals_304, primals_305, primals_306, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_320, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_339, primals_341, primals_342, primals_343, primals_344, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_358, primals_360, primals_361, primals_362, primals_363, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_377, primals_379, primals_380, primals_381, primals_382, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_396, primals_398, primals_399, primals_400, primals_401, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_415, primals_417, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_434, primals_436, primals_437, primals_438, primals_439, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_453, primals_455, primals_456, primals_457, primals_458, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_472, primals_474, primals_475, primals_476, primals_477, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_491, primals_493, primals_494, primals_495, primals_496, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_510, primals_512, primals_513, primals_514, primals_515, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_529, primals_531, primals_532, primals_533, primals_534, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_548, primals_550, primals_551, primals_552, primals_553, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_567, primals_569, primals_570, primals_571, primals_572, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_586, primals_588, primals_589, primals_590, primals_591, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_605, primals_607, primals_608, primals_609, primals_610, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_624, primals_626, primals_627, primals_628, primals_629, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_643, primals_645, primals_646, primals_647, primals_648, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_662, primals_664, primals_665, primals_666, primals_667, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_676, buf11, buf12, buf14, buf15, buf17, buf18, buf20, buf21, buf23, buf24, buf25, buf26, buf28, buf29, buf30, buf31, buf33, buf34, buf35, buf36, buf38, buf39, buf40, buf41, buf43, buf44, buf45, buf46, buf48, buf49, buf50, buf51, buf53, buf54, buf55, buf56, buf58, buf59, buf60, buf61, buf63, buf64, buf66, buf68, buf69, buf71, buf72, buf73, buf74, buf75, buf77, buf78, buf80, buf82, buf83, buf85, buf86, buf87, buf88, buf89, buf91, buf92, buf94, buf96, buf97, buf99, buf100, buf101, buf102, buf103, buf105, buf106, buf108, buf110, buf111, buf113, buf114, buf115, buf116, buf117, buf119, buf120, buf122, buf124, buf125, buf127, buf128, buf129, buf130, buf131, buf133, buf134, buf136, buf138, buf139, buf141, buf142, buf143, buf144, buf145, buf147, buf148, buf150, buf152, buf153, buf155, buf156, buf157, buf158, buf159, buf161, buf162, buf164, buf166, buf167, buf169, buf170, buf171, buf172, buf173, buf175, buf176, buf178, buf180, buf181, buf183, buf184, buf185, buf186, buf187, buf189, buf190, buf192, buf194, buf195, buf197, buf198, buf199, buf200, buf201, buf203, buf204, buf206, buf208, buf209, buf211, buf212, buf213, buf214, buf215, buf217, buf218, buf220, buf222, buf223, buf225, buf226, buf227, buf228, buf229, buf231, buf232, buf234, buf236, buf237, buf239, buf240, buf241, buf242, buf243, buf245, buf246, buf248, buf250, buf251, buf253, buf254, buf255, buf256, buf257, buf259, buf260, buf262, buf264, buf265, buf267, buf268, buf269, buf270, buf271, buf273, buf274, buf276, buf278, buf279, buf281, buf282, buf283, buf284, buf285, buf287, buf288, buf290, buf292, buf293, buf295, buf296, buf297, buf298, buf299, buf301, buf302, buf304, buf306, buf307, buf309, buf310, buf311, buf312, buf313, buf315, buf316, buf318, buf320, buf321, buf323, buf324, buf325, buf326, buf327, buf329, buf330, buf332, buf334, buf335, buf337, buf338, buf339, buf340, buf341, buf343, buf344, buf346, buf348, buf349, buf351, buf352, buf353, buf354, buf355, buf357, buf358, buf360, buf362, buf363, buf365, buf366, buf367, buf368, buf369, buf371, buf372, buf374, buf376, buf377, buf379, buf380, buf381, buf382, buf383, buf385, buf386, buf388, buf390, buf391, buf393, buf394, buf395, buf396, buf397, buf399, buf400, buf402, buf404, buf405, buf407, buf408, buf409, buf410, buf411, buf413, buf414, buf416, buf418, buf419, buf421, buf422, buf423, buf424, buf425, buf427, buf428, buf430, buf432, buf433, buf435, buf436, buf437, buf438, buf439, buf441, buf442, buf444, buf446, buf447, buf449, buf450, buf451, buf452, buf453, buf455, buf456, buf458, buf460, buf461, buf463, buf464, buf465, buf466, buf467, buf469, buf470, buf472, buf474, buf475, buf477, buf478, buf479, buf480, buf481, buf483, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((24, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((24, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((48, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((32, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((32, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((32, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((512, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((32, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((512, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((32, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((32, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((768, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((40, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((960, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((40, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((960, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((40, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((960, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((40, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((960, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((40, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((960, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((40, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((960, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((40, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((960, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((40, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((960, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((40, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((960, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((256, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((64, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((1536, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((1280, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((16, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
