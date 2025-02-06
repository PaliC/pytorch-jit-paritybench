# AOT ID: ['17_forward']
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


# kernel path: inductor_cache/pc/cpcgbbrbte4bzz7vtscc22nsnqhty52r53vo6qbg2xvx6x3mcghg.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   input_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton('triton_poi_fused_clone_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 4)
    x3 = xindex // 4
    y0 = (yindex % 4)
    y1 = yindex // 4
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x3 + 16*x2 + 64*y1), xmask & ymask)
    tl.store(out_ptr0 + (x5 + 16*y4), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/42/c42v6enidqxhafxseipfkezemrx3hoxpcvoebbyfbp4omd66dmuw.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.add, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_1 => add
#   input_2 => relu
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_4), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_add_relu_threshold_backward_1 = async_compile.triton('triton_poi_fused_add_relu_threshold_backward_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_relu_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_relu_threshold_backward_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/co/ccoi2v7lkwegkzqattgfjjsvkqnwxibuwhpfifllza6big2ofcuc.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute, %slice_2], -1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x3 = xindex // 80
    x4 = ((xindex // 5) % 16)
    x1 = ((xindex // 5) % 4)
    x2 = ((xindex // 20) % 4)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x0) + 64*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 5, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x4 + 64*x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x0 + 5*x2 + 20*x1 + 80*x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5j/c5jeibklwwvf5nunzlkqhy4jyukpiagrlztwz7usuqo353y5n4a2.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_5 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_6,), kwargs = {})
triton_poi_fused_relu_3 = async_compile.triton('triton_poi_fused_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 5)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b7/cb7yo7zsahztg2xpoefghjdmxqhokdmxxrbrzluukusepagkteoy.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute, %slice_3], -1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x3 = xindex // 96
    x4 = ((xindex // 6) % 16)
    x1 = ((xindex // 6) % 4)
    x2 = ((xindex // 24) % 4)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x0) + 64*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 6, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x4 + 16*((-4) + x0) + 64*x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x0 + 6*x2 + 24*x1 + 96*x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qd/cqd7sgn5izupf4ntgov5xnwa4ttvxipccy6yqvpqjukhz6hw5f3y.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_8 => relu_2
# Graph fragment:
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_15,), kwargs = {})
triton_poi_fused_relu_5 = async_compile.triton('triton_poi_fused_relu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 6)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dw/cdw3s4hxv6ffhlxrxu7zldbeo5c5xtzk3x2xeveqarqzcxiy6ssa.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   out => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_4, %view_13, %view_22], 2), kwargs = {})
triton_poi_fused_stack_6 = async_compile.triton('triton_poi_fused_stack_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 12)
    x0 = (xindex % 256)
    x2 = xindex // 3072
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 1024*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-4) + x1) + 1024*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 12, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 256*((-8) + x1) + 1024*x2), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (256, 4), (4, 1))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_7, (5, 5), (5, 1))
    assert_size_stride(primals_8, (5, ), (1, ))
    assert_size_stride(primals_9, (256, 5), (5, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (6, 6), (6, 1))
    assert_size_stride(primals_12, (6, ), (1, ))
    assert_size_stride(primals_13, (256, 6), (6, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(primals_1, buf0, 16, 16, grid=grid(16, 16), stream=stream0)
        buf1 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (64, 4), (4, 1), 0), reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf1)
        del primals_3
        buf2 = reinterpret_tensor(buf1, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf1  # reuse
        buf13 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.add, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_relu_threshold_backward_1.run(buf2, primals_4, buf13, 256, grid=grid(256), stream=stream0)
        del primals_4
        buf3 = empty_strided_cuda((64, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, reinterpret_tensor(buf2, (64, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 256), (1, 4), 0), alpha=1, beta=1, out=buf3)
        del primals_6
        buf4 = empty_strided_cuda((4, 4, 4, 5), (80, 20, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(primals_1, primals_2, buf4, 320, grid=grid(320), stream=stream0)
        buf5 = empty_strided_cuda((64, 5), (5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf4, (64, 5), (5, 1), 0), reinterpret_tensor(primals_7, (5, 5), (1, 5), 0), out=buf5)
        del primals_7
        buf6 = reinterpret_tensor(buf5, (4, 4, 4, 5), (80, 20, 5, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_3.run(buf6, primals_8, 320, grid=grid(320), stream=stream0)
        del primals_8
        buf7 = empty_strided_cuda((64, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf6, (64, 5), (5, 1), 0), reinterpret_tensor(primals_9, (5, 256), (1, 5), 0), alpha=1, beta=1, out=buf7)
        del primals_10
        buf8 = empty_strided_cuda((4, 4, 4, 6), (96, 24, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(primals_1, primals_2, buf8, 384, grid=grid(384), stream=stream0)
        del primals_1
        del primals_2
        buf9 = empty_strided_cuda((64, 6), (6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf8, (64, 6), (6, 1), 0), reinterpret_tensor(primals_11, (6, 6), (1, 6), 0), out=buf9)
        del primals_11
        buf10 = reinterpret_tensor(buf9, (4, 4, 4, 6), (96, 24, 6, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf10, primals_12, 384, grid=grid(384), stream=stream0)
        del primals_12
        buf11 = empty_strided_cuda((64, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, reinterpret_tensor(buf10, (64, 6), (6, 1), 0), reinterpret_tensor(primals_13, (6, 256), (1, 6), 0), alpha=1, beta=1, out=buf11)
        del primals_14
        buf12 = empty_strided_cuda((4, 4, 12, 256), (12288, 3072, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_6.run(buf3, buf7, buf11, buf12, 49152, grid=grid(49152), stream=stream0)
        del buf11
        del buf3
        del buf7
    return (reinterpret_tensor(buf12, (4, 256, 3, 4, 4), (12288, 1, 1024, 256, 3072), 0), reinterpret_tensor(buf0, (64, 4), (4, 1), 0), reinterpret_tensor(buf2, (64, 4), (4, 1), 0), reinterpret_tensor(buf4, (64, 5), (5, 1), 0), reinterpret_tensor(buf6, (64, 5), (5, 1), 0), reinterpret_tensor(buf8, (64, 6), (6, 1), 0), reinterpret_tensor(buf10, (64, 6), (6, 1), 0), primals_13, primals_9, primals_5, buf13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((5, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((5, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((6, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
