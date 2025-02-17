# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/dp/cdp73itnud7eymxqvtbbslmjj5ogrwda5tnmqd7vclq3n7tuiwza.py
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
    size_hints={'y': 128, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 72
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


# kernel path: inductor_cache/zr/czrgnh5z5xqgtjxbub63jkz6cn6sasi5byau5jfdiintpg2houox.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [1, 2, 1, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_1 = async_compile.triton('triton_poi_fused_constant_pad_nd_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 8192}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4489
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex // 67
    x2 = (xindex % 67)
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-65) + x2 + 64*x3 + 4096*y4), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (y0 + 3*x5 + 13467*y1), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ao/cao6b6ft663juzqbsrep2xzoqzplnhvnd6wa5ffkrbulgu3teweq.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %primals_2, %primals_3, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/v3/cv3az7uile2l6ne6kci3kz2mfnyabtezbmfzmw676enkqmgnzhtf.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_3 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_4, %primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 24), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
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


# kernel path: inductor_cache/5a/c5agpmeo64542dkq56xfaxtbsv7y6cnqacw3yu4go67rwquzubg2.py
# Topologically Sorted Source Nodes: [input_4, add, input_5], Original ATen: [aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add => add
#   input_4 => convolution_2
#   input_5 => relu_1
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_1, %primals_6, %primals_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %relu), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
triton_poi_fused_add_convolution_relu_4 = async_compile.triton('triton_poi_fused_add_convolution_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/mh/cmhty62uv45lzj7wnpuggz2gignyu6f733cvne2v3vovactdncaa.py
# Topologically Sorted Source Nodes: [x_1, input_7, add_1, input_8], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_1 => add_1
#   input_7 => convolution_4
#   input_8 => relu_2
#   x_1 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_1, [0, 0, 0, 0, 0, 4], 0.0), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_3, %primals_10, %primals_11, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_4, %constant_pad_nd_1), kwargs = {})
#   %relu_2 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_relu_5 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_relu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 114688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 28)
    x1 = xindex // 28
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 24, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 24*x1), tmp5, other=0.0)
    tmp7 = tmp2 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/q7/cq7zsxxzvt4ty5uylwwiwtlqvua654ymxxflxchxxg5xi5edmr7c.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   h => constant_pad_nd_2
# Graph fragment:
#   %constant_pad_nd_2 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_2, [0, 2, 0, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_6 = async_compile.triton('triton_poi_fused_constant_pad_nd_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 952) % 34)
    x1 = ((xindex // 28) % 34)
    x3 = xindex // 32368
    x4 = (xindex % 952)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 32, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + 896*x2 + 28672*x3), tmp5 & xmask, other=0.0)
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xj/cxjj4e6lfzjpnz2uobgrt5kyh6icfdlcdhsthcjj7gt72j5l3cbs.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_9 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_2, %primals_12, %primals_13, [2, 2], [0, 0], [1, 1], False, [0, 0], 28), kwargs = {})
triton_poi_fused_convolution_7 = async_compile.triton('triton_poi_fused_convolution_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 28)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ra/craq2kyxzn26mgnfi3372p5ptbrkmghaqdtwnp7pugffxputapux.py
# Topologically Sorted Source Nodes: [x_2, x_3, input_10, add_2, input_11], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_2 => add_2
#   input_10 => convolution_6
#   input_11 => relu_3
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => constant_pad_nd_3
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_2, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %constant_pad_nd_3 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem, [0, 0, 0, 0, 0, 4], 0.0), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_5, %primals_14, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_6, %constant_pad_nd_3), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_2,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_8 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 16)
    x2 = xindex // 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 28, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 56*x1 + 1792*x2), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (28 + x0 + 56*x1 + 1792*x2), tmp5, other=0.0)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.load(in_ptr1 + (896 + x0 + 56*x1 + 1792*x2), tmp5, other=0.0)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tl.load(in_ptr1 + (924 + x0 + 56*x1 + 1792*x2), tmp5, other=0.0)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tmp2 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/7h/c7hoonemmtzxg442y67yfw3f5hv3r2hbqhbczk3rncvvlqkb4skr.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_12 => convolution_7
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %primals_16, %primals_17, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), kwargs = {})
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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


# kernel path: inductor_cache/xs/cxsmz67xush4k4t4ftmucra2nzidryywcivdkhiogy3mi477u7mz.py
# Topologically Sorted Source Nodes: [x_4, input_13, add_3, input_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_3 => add_3
#   input_13 => convolution_8
#   input_14 => relu_4
#   x_4 => constant_pad_nd_4
# Graph fragment:
#   %constant_pad_nd_4 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_3, [0, 0, 0, 0, 0, 4], 0.0), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_7, %primals_18, %primals_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %constant_pad_nd_4), kwargs = {})
#   %relu_4 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_relu_10 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_relu_10(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 36)
    x1 = xindex // 36
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 32, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 32*x1), tmp5, other=0.0)
    tmp7 = tmp2 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/ul/culo6q76nwkkiaoqlooyztmdq6upmbvgajkiv2spvtnww5az6glo.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_15 => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_20, %primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 36), kwargs = {})
triton_poi_fused_convolution_11 = async_compile.triton('triton_poi_fused_convolution_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 36)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/3x/c3x3kvdc7dph33bjc2bxxk6pkd7vzmqz63aa3oer3ukueag3zmeu.py
# Topologically Sorted Source Nodes: [x_5, input_16, add_4, input_17], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_4 => add_4
#   input_16 => convolution_10
#   input_17 => relu_5
#   x_5 => constant_pad_nd_5
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_4, [0, 0, 0, 0, 0, 6], 0.0), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_9, %primals_22, %primals_23, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %constant_pad_nd_5), kwargs = {})
#   %relu_5 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_relu_12 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_relu_12(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 42)
    x1 = xindex // 42
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 36, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 36*x1), tmp5 & xmask, other=0.0)
    tmp7 = tmp2 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sw/cswkfjsehgf5dxcj2eo6zwthe5mux3lt7ervdwvr6s6awon5ezwr.py
# Topologically Sorted Source Nodes: [h_1], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   h_1 => constant_pad_nd_6
# Graph fragment:
#   %constant_pad_nd_6 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_5, [0, 2, 0, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_13 = async_compile.triton('triton_poi_fused_constant_pad_nd_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 54432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 756) % 18)
    x1 = ((xindex // 42) % 18)
    x3 = xindex // 13608
    x4 = (xindex % 756)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + 672*x2 + 10752*x3), tmp5 & xmask, other=0.0)
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bg/cbgcv4aos23hbl6uqfxqm4aotnshweqkh2n3kmbilytnfurrv6f3.py
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_18 => convolution_11
# Graph fragment:
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_6, %primals_24, %primals_25, [2, 2], [0, 0], [1, 1], False, [0, 0], 42), kwargs = {})
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 42)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cv/ccvzspk7r5zp6bzat46g6rigoyet4w24jmk5e4jwrnezmrrxzax2.py
# Topologically Sorted Source Nodes: [x_6, x_7, input_19, add_5, input_20], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_5 => add_5
#   input_19 => convolution_12
#   input_20 => relu_6
#   x_6 => _low_memory_max_pool2d_with_offsets_1
#   x_7 => constant_pad_nd_7
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %constant_pad_nd_7 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_2, [0, 0, 0, 0, 0, 6], 0.0), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_11, %primals_26, %primals_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %constant_pad_nd_7), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_15 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 48)
    x1 = ((xindex // 48) % 8)
    x2 = xindex // 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 42, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 84*x1 + 1344*x2), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (42 + x0 + 84*x1 + 1344*x2), tmp5, other=0.0)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.load(in_ptr1 + (672 + x0 + 84*x1 + 1344*x2), tmp5, other=0.0)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tl.load(in_ptr1 + (714 + x0 + 84*x1 + 1344*x2), tmp5, other=0.0)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tmp2 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/td/ctddzckwsrrhppi2szz6lwj64cnjlq3nlpnx3q2i7e2tpeuwlaa3.py
# Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_21 => convolution_13
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %primals_28, %primals_29, [1, 1], [1, 1], [1, 1], False, [0, 0], 48), kwargs = {})
triton_poi_fused_convolution_16 = async_compile.triton('triton_poi_fused_convolution_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 48)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/fy/cfyr6lti36m34ijkgglbj6bybevogbqeemxopkc46l2lsadfbgs7.py
# Topologically Sorted Source Nodes: [x_8, input_22, add_6, input_23], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_6 => add_6
#   input_22 => convolution_14
#   input_23 => relu_7
#   x_8 => constant_pad_nd_8
# Graph fragment:
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_6, [0, 0, 0, 0, 0, 8], 0.0), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_13, %primals_30, %primals_31, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %constant_pad_nd_8), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_relu_17 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_relu_17(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 56)
    x1 = xindex // 56
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 48, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 48*x1), tmp5 & xmask, other=0.0)
    tmp7 = tmp2 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fn/cfnpyiztmfyoermwlullvd5vcdiijcgd4wgofjoi2yo4rpvf2k4t.py
# Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_24 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_32, %primals_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 56), kwargs = {})
triton_poi_fused_convolution_18 = async_compile.triton('triton_poi_fused_convolution_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 56)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fu/cfujyp7zmycgtynuaaduuzyy6spjdajyuo7imsvaar6esaa34lla.py
# Topologically Sorted Source Nodes: [x_9, input_25, add_7, input_26], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_7 => add_7
#   input_25 => convolution_16
#   input_26 => relu_8
#   x_9 => constant_pad_nd_9
# Graph fragment:
#   %constant_pad_nd_9 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_7, [0, 0, 0, 0, 0, 8], 0.0), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_15, %primals_34, %primals_35, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_16, %constant_pad_nd_9), kwargs = {})
#   %relu_8 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_relu_19 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_relu_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_relu_19(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 56, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 56*x1), tmp5, other=0.0)
    tmp7 = tmp2 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/7u/c7unpihzpxbuq246x3zrjdftmuyj5fuh3qoxy4tuuyctrjupbw7n.py
# Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_27 => convolution_17
# Graph fragment:
#   %convolution_17 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_36, %primals_37, [1, 1], [1, 1], [1, 1], False, [0, 0], 64), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/4c/c4cjgvdemvxtdury7yzoey27s5yezc5gvcrzjqog4lqtfswf3zpc.py
# Topologically Sorted Source Nodes: [x_10, input_28, add_8, input_29], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_8 => add_8
#   input_28 => convolution_18
#   input_29 => relu_9
#   x_10 => constant_pad_nd_10
# Graph fragment:
#   %constant_pad_nd_10 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_8, [0, 0, 0, 0, 0, 8], 0.0), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_17, %primals_38, %primals_39, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_18, %constant_pad_nd_10), kwargs = {})
#   %relu_9 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_relu_21 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_relu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_relu_21(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 72)
    x1 = xindex // 72
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 64, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 64*x1), tmp5 & xmask, other=0.0)
    tmp7 = tmp2 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zw/czwrtbifmid5ul3bbfyohum3qpdlr6szqi3nupeykedgjstqvfn7.py
# Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_30 => convolution_19
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %primals_40, %primals_41, [1, 1], [1, 1], [1, 1], False, [0, 0], 72), kwargs = {})
triton_poi_fused_convolution_22 = async_compile.triton('triton_poi_fused_convolution_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 72)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ke/ckefbor3lxng3gmon7xa56fke7456pdmutrbhn2dp7r4453u2d2m.py
# Topologically Sorted Source Nodes: [x_11, input_31, add_9, input_32], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_9 => add_9
#   input_31 => convolution_20
#   input_32 => relu_10
#   x_11 => constant_pad_nd_11
# Graph fragment:
#   %constant_pad_nd_11 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_9, [0, 0, 0, 0, 0, 8], 0.0), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_19, %primals_42, %primals_43, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_20, %constant_pad_nd_11), kwargs = {})
#   %relu_10 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_relu_23 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_relu_23(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 80)
    x1 = xindex // 80
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 72, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 72*x1), tmp5, other=0.0)
    tmp7 = tmp2 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/dd/cddmyeeaubpqjfns7tabewlgqjbhs5y4m7k7m4t44omb764knwyg.py
# Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_33 => convolution_21
# Graph fragment:
#   %convolution_21 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %primals_44, %primals_45, [1, 1], [1, 1], [1, 1], False, [0, 0], 80), kwargs = {})
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 80)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/mb/cmb3evftdlwyebrpd2lgdsrsl3dhsrkeogouppr4jqfnytgpjc7q.py
# Topologically Sorted Source Nodes: [x_12, input_34, add_10, input_35], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_10 => add_10
#   input_34 => convolution_22
#   input_35 => relu_11
#   x_12 => constant_pad_nd_12
# Graph fragment:
#   %constant_pad_nd_12 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_10, [0, 0, 0, 0, 0, 8], 0.0), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_21, %primals_46, %primals_47, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %constant_pad_nd_12), kwargs = {})
#   %relu_11 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_relu_25 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_relu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_relu_25(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 88)
    x1 = xindex // 88
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 80, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 80*x1), tmp5 & xmask, other=0.0)
    tmp7 = tmp2 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zj/czjjvbtitvbfsmxktnlskcfrfvzxsvaqgpe3l6ie2pu5v34zomv6.py
# Topologically Sorted Source Nodes: [h_2], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   h_2 => constant_pad_nd_13
# Graph fragment:
#   %constant_pad_nd_13 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_11, [0, 2, 0, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_26 = async_compile.triton('triton_poi_fused_constant_pad_nd_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 35200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 880) % 10)
    x1 = ((xindex // 88) % 10)
    x3 = xindex // 8800
    x4 = (xindex % 880)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 8, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + 704*x2 + 5632*x3), tmp5 & xmask, other=0.0)
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/px/cpxb2ocponf2z7ewn2f3dgsfkvq5ccirzedwakzosowyha2w6nps.py
# Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_36 => convolution_23
# Graph fragment:
#   %convolution_23 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_13, %primals_48, %primals_49, [2, 2], [0, 0], [1, 1], False, [0, 0], 88), kwargs = {})
triton_poi_fused_convolution_27 = async_compile.triton('triton_poi_fused_convolution_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_27(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 88)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/c2/cc2357amrzr6eqytb7z4hk5aizsuzuf7alzblhyw7hc3tzb4zjuc.py
# Topologically Sorted Source Nodes: [x_13, x_14, input_37, add_11, input_38], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_11 => add_11
#   input_37 => convolution_24
#   input_38 => relu_12
#   x_13 => _low_memory_max_pool2d_with_offsets_2
#   x_14 => constant_pad_nd_14
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %constant_pad_nd_14 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_4, [0, 0, 0, 0, 0, 8], 0.0), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_23, %primals_50, %primals_51, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_24, %constant_pad_nd_14), kwargs = {})
#   %relu_12 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_28 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_28(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 4)
    x2 = xindex // 384
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 88, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + 176*x1 + 1408*x2), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (88 + x0 + 176*x1 + 1408*x2), tmp5 & xmask, other=0.0)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.load(in_ptr1 + (704 + x0 + 176*x1 + 1408*x2), tmp5 & xmask, other=0.0)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tl.load(in_ptr1 + (792 + x0 + 176*x1 + 1408*x2), tmp5 & xmask, other=0.0)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tmp2 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/re/crevjpvsseyij76lt3k42iipjb4l3qdk5dfujhs4maje56ex75c3.py
# Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_39 => convolution_25
# Graph fragment:
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %primals_52, %primals_53, [1, 1], [1, 1], [1, 1], False, [0, 0], 96), kwargs = {})
triton_poi_fused_convolution_29 = async_compile.triton('triton_poi_fused_convolution_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_29(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wk/cwk2sxmx2tmp3r32pxq6x63aietcox4ve2ma7re6umv36qbeaeiv.py
# Topologically Sorted Source Nodes: [input_40, add_12, input_41], Original ATen: [aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_12 => add_12
#   input_40 => convolution_26
#   input_41 => relu_13
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_25, %primals_54, %primals_55, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %relu_12), kwargs = {})
#   %relu_13 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
triton_poi_fused_add_convolution_relu_30 = async_compile.triton('triton_poi_fused_add_convolution_relu_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_30(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mq/cmqc6kuqunkt5hxhrkiobv32hgfgou46qfqi4whxezggycjz3dbg.py
# Topologically Sorted Source Nodes: [c], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   c => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %view_1], 1), kwargs = {})
triton_poi_fused_cat_31 = async_compile.triton('triton_poi_fused_cat_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 224)
    x1 = xindex // 224
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x1 + (((x0) % 128))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (((x0) % 2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 224, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (96*x1 + ((((-128) + x0) % 96))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + ((((-128) + x0) % 6)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/45/c45wunit75fbbtemjlpukycoop6nqdlgdtd2i736mwvhpwqi24mz.py
# Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_13 => getitem_5
# Graph fragment:
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_32 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 88)
    x1 = ((xindex // 88) % 4)
    x2 = xindex // 352
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 176*x1 + 1408*x2), xmask)
    tmp1 = tl.load(in_ptr0 + (88 + x0 + 176*x1 + 1408*x2), xmask)
    tmp7 = tl.load(in_ptr0 + (704 + x0 + 176*x1 + 1408*x2), xmask)
    tmp12 = tl.load(in_ptr0 + (792 + x0 + 176*x1 + 1408*x2), xmask)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qu/cquwzjux752ghr2uwny36kfyzzcksx5mwht4e5yf5nl3fvvglt53.py
# Topologically Sorted Source Nodes: [r], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   r => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_2, %view_3], 1), kwargs = {})
triton_poi_fused_cat_33 = async_compile.triton('triton_poi_fused_cat_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 224)
    x0 = (xindex % 16)
    x2 = xindex // 3584
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*((((x0 + 16*(x1) + 2048*x2) // 2048) % 4)) + (((x0 + 16*(x1)) % 2048))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (((x0 + 16*(x1)) % 32)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 224, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (1536*((((x0 + 16*((-128) + x1) + 1536*x2) // 1536) % 4)) + (((x0 + 16*((-128) + x1)) % 1536))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (((x0 + 16*((-128) + x1)) % 96)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2n/c2nnkalmvbz2mv72rbmwxolriy2dfyi3adbd5wu7d3oipsonvcxk.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_6 => getitem_3
# Graph fragment:
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_34 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 42)
    x1 = ((xindex // 42) % 8)
    x2 = xindex // 336
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 84*x1 + 1344*x2), xmask)
    tmp1 = tl.load(in_ptr0 + (42 + x0 + 84*x1 + 1344*x2), xmask)
    tmp7 = tl.load(in_ptr0 + (672 + x0 + 84*x1 + 1344*x2), xmask)
    tmp12 = tl.load(in_ptr0 + (714 + x0 + 84*x1 + 1344*x2), xmask)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4b/c4b2bzrw32nmj5cqw2nspw3a4mz5poznroi67wctcrrj7upwjvup.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_2 => getitem_1
# Graph fragment:
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_35 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 28)
    x1 = ((xindex // 28) % 16)
    x2 = xindex // 448
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 56*x1 + 1792*x2), None)
    tmp1 = tl.load(in_ptr0 + (28 + x0 + 56*x1 + 1792*x2), None)
    tmp7 = tl.load(in_ptr0 + (896 + x0 + 56*x1 + 1792*x2), None)
    tmp12 = tl.load(in_ptr0 + (924 + x0 + 56*x1 + 1792*x2), None)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (24, 3, 5, 5), (75, 25, 5, 1))
    assert_size_stride(primals_3, (24, ), (1, ))
    assert_size_stride(primals_4, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_7, (24, ), (1, ))
    assert_size_stride(primals_8, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_9, (24, ), (1, ))
    assert_size_stride(primals_10, (28, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_11, (28, ), (1, ))
    assert_size_stride(primals_12, (28, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_13, (28, ), (1, ))
    assert_size_stride(primals_14, (32, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (36, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_19, (36, ), (1, ))
    assert_size_stride(primals_20, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_21, (36, ), (1, ))
    assert_size_stride(primals_22, (42, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(primals_23, (42, ), (1, ))
    assert_size_stride(primals_24, (42, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_25, (42, ), (1, ))
    assert_size_stride(primals_26, (48, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_27, (48, ), (1, ))
    assert_size_stride(primals_28, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_29, (48, ), (1, ))
    assert_size_stride(primals_30, (56, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_31, (56, ), (1, ))
    assert_size_stride(primals_32, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_33, (56, ), (1, ))
    assert_size_stride(primals_34, (64, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (72, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_39, (72, ), (1, ))
    assert_size_stride(primals_40, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_41, (72, ), (1, ))
    assert_size_stride(primals_42, (80, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_43, (80, ), (1, ))
    assert_size_stride(primals_44, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_45, (80, ), (1, ))
    assert_size_stride(primals_46, (88, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_47, (88, ), (1, ))
    assert_size_stride(primals_48, (88, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_49, (88, ), (1, ))
    assert_size_stride(primals_50, (96, 88, 1, 1), (88, 1, 1, 1))
    assert_size_stride(primals_51, (96, ), (1, ))
    assert_size_stride(primals_52, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_53, (96, ), (1, ))
    assert_size_stride(primals_54, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_55, (96, ), (1, ))
    assert_size_stride(primals_56, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_57, (96, ), (1, ))
    assert_size_stride(primals_58, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_59, (96, ), (1, ))
    assert_size_stride(primals_60, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_61, (96, ), (1, ))
    assert_size_stride(primals_62, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_63, (96, ), (1, ))
    assert_size_stride(primals_64, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_65, (96, ), (1, ))
    assert_size_stride(primals_66, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_67, (96, ), (1, ))
    assert_size_stride(primals_68, (2, 88, 1, 1), (88, 1, 1, 1))
    assert_size_stride(primals_69, (2, ), (1, ))
    assert_size_stride(primals_70, (6, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_71, (6, ), (1, ))
    assert_size_stride(primals_72, (32, 88, 1, 1), (88, 1, 1, 1))
    assert_size_stride(primals_73, (32, ), (1, ))
    assert_size_stride(primals_74, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_75, (96, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((24, 3, 5, 5), (75, 1, 15, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_2, buf0, 72, 25, grid=grid(72, 25), stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((4, 3, 67, 67), (13467, 1, 201, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_1.run(primals_1, buf1, 12, 4489, grid=grid(12, 4489), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf3, primals_3, 98304, grid=grid(98304), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf4, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf5, primals_5, 98304, grid=grid(98304), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_4, add, input_5], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_4.run(buf7, primals_7, buf3, 98304, grid=grid(98304), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf8, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf9, primals_9, 98304, grid=grid(98304), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 28, 32, 32), (28672, 1, 896, 28))
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_1, input_7, add_1, input_8], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_relu_5.run(buf11, primals_11, buf7, 114688, grid=grid(114688), stream=stream0)
        del primals_11
        buf12 = empty_strided_cuda((4, 28, 34, 34), (32368, 1, 952, 28), torch.float32)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_6.run(buf11, buf12, 129472, grid=grid(129472), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf12, primals_12, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=28, bias=None)
        assert_size_stride(buf14, (4, 28, 16, 16), (7168, 1, 448, 28))
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_7.run(buf15, primals_13, 28672, grid=grid(28672), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3, input_10, add_2, input_11], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_8.run(buf17, primals_15, buf11, 32768, grid=grid(32768), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf18, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf19, primals_17, 32768, grid=grid(32768), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 36, 16, 16), (9216, 1, 576, 36))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_4, input_13, add_3, input_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_relu_10.run(buf21, primals_19, buf17, 36864, grid=grid(36864), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
        assert_size_stride(buf22, (4, 36, 16, 16), (9216, 1, 576, 36))
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_11.run(buf23, primals_21, 36864, grid=grid(36864), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 42, 16, 16), (10752, 1, 672, 42))
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_5, input_16, add_4, input_17], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_relu_12.run(buf25, primals_23, buf21, 43008, grid=grid(43008), stream=stream0)
        del primals_23
        buf26 = empty_strided_cuda((4, 42, 18, 18), (13608, 1, 756, 42), torch.float32)
        # Topologically Sorted Source Nodes: [h_1], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_13.run(buf25, buf26, 54432, grid=grid(54432), stream=stream0)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf26, primals_24, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=42, bias=None)
        assert_size_stride(buf28, (4, 42, 8, 8), (2688, 1, 336, 42))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_14.run(buf29, primals_25, 10752, grid=grid(10752), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 48, 8, 8), (3072, 1, 384, 48))
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7, input_19, add_5, input_20], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_15.run(buf31, primals_27, buf25, 12288, grid=grid(12288), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf32, (4, 48, 8, 8), (3072, 1, 384, 48))
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_16.run(buf33, primals_29, 12288, grid=grid(12288), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 56, 8, 8), (3584, 1, 448, 56))
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_8, input_22, add_6, input_23], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_relu_17.run(buf35, primals_31, buf31, 14336, grid=grid(14336), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf36, (4, 56, 8, 8), (3584, 1, 448, 56))
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_18.run(buf37, primals_33, 14336, grid=grid(14336), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_9, input_25, add_7, input_26], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_relu_19.run(buf39, primals_35, buf35, 16384, grid=grid(16384), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf40, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf41, primals_37, 16384, grid=grid(16384), stream=stream0)
        del primals_37
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 72, 8, 8), (4608, 1, 576, 72))
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_10, input_28, add_8, input_29], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_relu_21.run(buf43, primals_39, buf39, 18432, grid=grid(18432), stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf44, (4, 72, 8, 8), (4608, 1, 576, 72))
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf45, primals_41, 18432, grid=grid(18432), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 80, 8, 8), (5120, 1, 640, 80))
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_11, input_31, add_9, input_32], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_relu_23.run(buf47, primals_43, buf43, 20480, grid=grid(20480), stream=stream0)
        del primals_43
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf48, (4, 80, 8, 8), (5120, 1, 640, 80))
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf49, primals_45, 20480, grid=grid(20480), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 88, 8, 8), (5632, 1, 704, 88))
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_12, input_34, add_10, input_35], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_relu_25.run(buf51, primals_47, buf47, 22528, grid=grid(22528), stream=stream0)
        del primals_47
        buf52 = empty_strided_cuda((4, 88, 10, 10), (8800, 1, 880, 88), torch.float32)
        # Topologically Sorted Source Nodes: [h_2], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_26.run(buf51, buf52, 35200, grid=grid(35200), stream=stream0)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf52, primals_48, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=88, bias=None)
        assert_size_stride(buf54, (4, 88, 4, 4), (1408, 1, 352, 88))
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_27.run(buf55, primals_49, 5632, grid=grid(5632), stream=stream0)
        del primals_49
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_13, x_14, input_37, add_11, input_38], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_max_pool2d_with_indices_relu_28.run(buf57, primals_51, buf51, 6144, grid=grid(6144), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf58, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf59, primals_53, 6144, grid=grid(6144), stream=stream0)
        del primals_53
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_40, add_12, input_41], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_30.run(buf61, primals_55, buf57, 6144, grid=grid(6144), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf62, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf63, primals_57, 6144, grid=grid(6144), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [input_43, add_13, input_44], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_30.run(buf65, primals_59, buf61, 6144, grid=grid(6144), stream=stream0)
        del primals_59
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf66, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf67, primals_61, 6144, grid=grid(6144), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [input_46, add_14, input_47], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_30.run(buf69, primals_63, buf65, 6144, grid=grid(6144), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf70, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf71, primals_65, 6144, grid=grid(6144), stream=stream0)
        del primals_65
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_49, add_15, input_50], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_30.run(buf73, primals_67, buf69, 6144, grid=grid(6144), stream=stream0)
        del primals_67
        # Topologically Sorted Source Nodes: [c2], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf73, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 6, 4, 4), (96, 1, 24, 6))
        # Topologically Sorted Source Nodes: [c1], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf51, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 2, 8, 8), (128, 1, 16, 2))
        buf76 = empty_strided_cuda((4, 224, 1), (224, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_31.run(buf74, primals_69, buf75, primals_71, buf76, 896, grid=grid(896), stream=stream0)
        del buf74
        del buf75
        del primals_69
        del primals_71
        # Topologically Sorted Source Nodes: [r2], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf73, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf53 = empty_strided_cuda((4, 88, 4, 4), (1408, 1, 352, 88), torch.int8)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_32.run(buf51, buf53, 5632, grid=grid(5632), stream=stream0)
        # Topologically Sorted Source Nodes: [r1], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf51, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf79 = empty_strided_cuda((4, 224, 16), (3584, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [r], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_33.run(buf77, primals_73, buf78, primals_75, buf79, 14336, grid=grid(14336), stream=stream0)
        del buf77
        del buf78
        del primals_73
        del primals_75
        buf27 = empty_strided_cuda((4, 42, 8, 8), (2688, 1, 336, 42), torch.int8)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_34.run(buf25, buf27, 10752, grid=grid(10752), stream=stream0)
        buf13 = empty_strided_cuda((4, 28, 16, 16), (7168, 1, 448, 28), torch.int8)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_35.run(buf11, buf13, 28672, grid=grid(28672), stream=stream0)
    return (buf79, buf76, buf0, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, buf1, buf3, buf5, buf7, buf9, buf11, buf12, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf26, buf27, buf29, buf31, buf33, buf35, buf37, buf39, buf41, buf43, buf45, buf47, buf49, buf51, buf52, buf53, buf55, buf57, buf59, buf61, buf63, buf65, buf67, buf69, buf71, buf73, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((24, 3, 5, 5), (75, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((28, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((28, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((36, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((42, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((42, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((48, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((56, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((72, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((80, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((88, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((88, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((96, 88, 1, 1), (88, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((2, 88, 1, 1), (88, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((6, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((32, 88, 1, 1), (88, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
