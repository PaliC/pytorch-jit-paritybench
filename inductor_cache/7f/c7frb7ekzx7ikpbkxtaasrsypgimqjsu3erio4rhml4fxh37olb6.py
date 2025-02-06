# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/n3/cn3ajxnaipai7oxilqgska7ekg4o3ynxnzmcjkx2kbvrekhacq5c.py
# Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   out_16 => iota
# Graph fragment:
#   %iota : [num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
triton_poi_fused_reflection_pad2d_0 = async_compile.triton('triton_poi_fused_reflection_pad2d_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eh/ceh76uaegjotboz2owrucee322nsqcjqqhokb2evmqy6oxcsalzo.py
# Topologically Sorted Source Nodes: [index_select], Original ATen: [aten.index_select]
# Source node to ATen node mapping:
#   index_select => index
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%primals_12, [None, %primals_13]), kwargs = {})
triton_poi_fused_index_select_1 = async_compile.triton('triton_poi_fused_index_select_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 7, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 7)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 7")
    tmp6 = tl.load(in_ptr1 + (tmp4 + 7*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tu/ctuqc37gmupdfuwv4vomeyiaou2uutiwl6o4vwtpqnpsl22hodlh.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 8}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    x3 = xindex // 4
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x4 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + 8*y5), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/3c/c3cbsqrbcnhqby44r4f2tj4zezcxjpkcblhtqaj4cuesigstxxl7.py
# Topologically Sorted Source Nodes: [out_9, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_10 => relu_3
#   out_9 => add_18, mul_31, mul_32, sub_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_1), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_3), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_5), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_7), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_18,), kwargs = {})
#   %le_48 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gi/cgilgtecj5b2nyps63tqqhlacclggtlx454nckafmtthgeuxmtch.py
# Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_2 => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_43,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 8*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/25/c25k2mowsbweprygh3ctse56bqp7mnuc6x5klus65qj2rcxnvelc.py
# Topologically Sorted Source Nodes: [out_19], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   out_19 => _unsafe_index_2, _unsafe_index_3
# Graph fragment:
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_42, [None, None, %sub_21, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %sub_21]), kwargs = {})
triton_poi_fused_reflection_pad2d_5 = async_compile.triton('triton_poi_fused_reflection_pad2d_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7n/c7n54rhc3caxiuh2ec7elzummaakendeebb2msepqjvfaoyzcbnf.py
# Topologically Sorted Source Nodes: [out_20, out_21, out_23, out_24], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_20 => convolution_9
#   out_21 => relu_7
#   out_23 => convolution_10
#   out_24 => relu_8
# Graph fragment:
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_3, %primals_45, %primals_46, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_5, %primals_45, %primals_46, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %le_41 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
#   %le_42 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_7, 0), kwargs = {})
triton_poi_fused_convolution_relu_threshold_backward_6 = async_compile.triton('triton_poi_fused_convolution_relu_threshold_backward_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tmp8 = tmp7 + tmp1
    tmp9 = triton_helpers.maximum(tmp3, tmp8)
    tmp10 = tmp9 <= tmp5
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/py/cpybxh5adxur6rhv3phgb7o6zwegzj7ehuu2hdiojxy252ycprjr.py
# Topologically Sorted Source Nodes: [out_25, out_26], Original ATen: [aten.cat, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   out_25 => cat_9
#   out_26 => _unsafe_index_6, _unsafe_index_7
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_7, %relu_8], 1), kwargs = {})
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%cat_9, [None, None, %sub_21, None]), kwargs = {})
#   %_unsafe_index_7 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_6, [None, None, None, %sub_21]), kwargs = {})
triton_poi_fused_cat_reflection_pad2d_7 = async_compile.triton('triton_poi_fused_cat_reflection_pad2d_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_reflection_pad2d_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_reflection_pad2d_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 8)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x3 = xindex // 128
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + x1)) + 16*(x2) + 64*x3), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 8, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + x1)) + 16*((-4) + x2) + 64*x3), tmp12 & xmask, other=0.0)
    tmp16 = tl.load(in_ptr1 + ((-4) + x2), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x5), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zv/czvnjv4mcyynt45mfd733y46ri366ytbppcb37g3yxc24lk7anhl.py
# Topologically Sorted Source Nodes: [out_27, out_28], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_27 => convolution_11
#   out_28 => relu_9
# Graph fragment:
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_7, %primals_47, %primals_48, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %le_38 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
triton_poi_fused_convolution_relu_threshold_backward_8 = async_compile.triton('triton_poi_fused_convolution_relu_threshold_backward_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zx/czxmypmcqdjse6ucowov7mvjlngfaioztsx7scowspw6wz4q5jip.py
# Topologically Sorted Source Nodes: [out_27, out_28, out_29], Original ATen: [aten.convolution, aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   out_27 => convolution_11
#   out_28 => relu_9
#   out_29 => _unsafe_index_8, _unsafe_index_9
# Graph fragment:
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_7, %primals_47, %primals_48, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_9, [None, None, %sub_37, None]), kwargs = {})
#   %_unsafe_index_9 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_8, [None, None, None, %sub_37]), kwargs = {})
triton_poi_fused_convolution_reflection_pad2d_relu_9 = async_compile.triton('triton_poi_fused_convolution_reflection_pad2d_relu_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_reflection_pad2d_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_reflection_pad2d_relu_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = ((xindex // 6) % 6)
    x4 = xindex // 36
    x2 = ((xindex // 36) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x5), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qj/cqj6xitc6kyhceruewwhxkoobcr2xzkyenhhvaapaxefl4ultipe.py
# Topologically Sorted Source Nodes: [qkv, qkv_2], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   qkv => add_3, mul_4, mul_5, sub_1
#   qkv_2 => add_20, mul_34, mul_35, sub_11
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_8), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_10), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_11), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_8), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_9), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_10), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
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
    tmp17 = tmp16 - tmp1
    tmp18 = tmp17 * tmp10
    tmp19 = tmp18 * tmp12
    tmp20 = tmp19 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lo/clokmomnqycr7a2u6rridssl6tvcxjcswrkckypw54ddwc5qfpsu.py
# Topologically Sorted Source Nodes: [stacked_similarity, stacked_similarity_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   stacked_similarity => cat
#   stacked_similarity_4 => cat_4
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_5, %view_3, %permute_6], 1), kwargs = {})
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_39, %view_37, %permute_48], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 3)
    x1 = ((xindex // 4) % 4)
    x3 = xindex // 48
    x0 = (xindex % 4)
    x6 = (xindex % 16)
    x7 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + 16*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (4 + x0 + 16*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 2, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr0 + (x1 + 16*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr1 + (x6), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 3, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (4 + x0 + 16*x3), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr1 + (16 + x1 + 4*x0), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp29 = tl.load(in_ptr2 + (x1 + 16*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr2 + (4 + x0 + 16*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tl.load(in_ptr2 + (x1 + 16*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 * tmp15
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp13, tmp35, tmp36)
    tmp38 = tl.load(in_ptr2 + (4 + x0 + 16*x3), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38 * tmp23
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp19, tmp39, tmp40)
    tmp42 = tl.where(tmp13, tmp37, tmp41)
    tmp43 = tl.where(tmp4, tmp33, tmp42)
    tl.store(out_ptr0 + (x7), tmp28, xmask)
    tl.store(out_ptr1 + (x7), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fs/cfsflvpgmyxaw2nypmtozzogj3keflb2kiyhslzqsotddmt6xxt6.py
# Topologically Sorted Source Nodes: [stacked_similarity_1, stacked_similarity_5], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   stacked_similarity_1 => sum_1
#   stacked_similarity_5 => sum_7
# Graph fragment:
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_6, [1]), kwargs = {})
#   %sum_7 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_40, [1]), kwargs = {})
triton_poi_fused_sum_12 = async_compile.triton('triton_poi_fused_sum_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 48*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (16 + x0 + 48*x1), xmask)
    tmp21 = tl.load(in_ptr1 + (1))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp24 = tl.load(in_ptr2 + (1))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp31 = tl.load(in_ptr3 + (1))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp34 = tl.load(in_ptr4 + (1))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp38 = tl.load(in_ptr0 + (32 + x0 + 48*x1), xmask)
    tmp39 = tl.load(in_ptr1 + (2))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp42 = tl.load(in_ptr2 + (2))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp49 = tl.load(in_ptr3 + (2))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp52 = tl.load(in_ptr4 + (2))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp56 = tl.load(in_ptr5 + (x0 + 48*x1), xmask)
    tmp61 = tl.load(in_ptr5 + (16 + x0 + 48*x1), xmask)
    tmp67 = tl.load(in_ptr5 + (32 + x0 + 48*x1), xmask)
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp23 = tmp20 - tmp22
    tmp26 = tmp25 + tmp6
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = tmp9 / tmp27
    tmp29 = tmp28 * tmp11
    tmp30 = tmp23 * tmp29
    tmp33 = tmp30 * tmp32
    tmp36 = tmp33 + tmp35
    tmp37 = tmp19 + tmp36
    tmp41 = tmp38 - tmp40
    tmp44 = tmp43 + tmp6
    tmp45 = libdevice.sqrt(tmp44)
    tmp46 = tmp9 / tmp45
    tmp47 = tmp46 * tmp11
    tmp48 = tmp41 * tmp47
    tmp51 = tmp48 * tmp50
    tmp54 = tmp51 + tmp53
    tmp55 = tmp37 + tmp54
    tmp57 = tmp56 - tmp2
    tmp58 = tmp57 * tmp12
    tmp59 = tmp58 * tmp15
    tmp60 = tmp59 + tmp18
    tmp62 = tmp61 - tmp22
    tmp63 = tmp62 * tmp29
    tmp64 = tmp63 * tmp32
    tmp65 = tmp64 + tmp35
    tmp66 = tmp60 + tmp65
    tmp68 = tmp67 - tmp40
    tmp69 = tmp68 * tmp47
    tmp70 = tmp69 * tmp50
    tmp71 = tmp70 + tmp53
    tmp72 = tmp66 + tmp71
    tl.store(out_ptr0 + (x2), tmp55, xmask)
    tl.store(out_ptr1 + (x2), tmp72, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sm/csml3sfmtn7b3dwdqfhc2r2rb3rcv2vkhrzl4tihlnnd2m4jqae3.py
# Topologically Sorted Source Nodes: [similarity], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   similarity => amax, exp, sub_3
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%sum_1, [3], True), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
triton_poi_fused__softmax_13 = async_compile.triton('triton_poi_fused__softmax_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uy/cuycc3xxwykq2eqqjkt6iabezdhhwcq2oiidqdkiop4rkbctswci.py
# Topologically Sorted Source Nodes: [similarity], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   similarity => div, sum_2
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [3], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_2), kwargs = {})
triton_poi_fused__softmax_14 = async_compile.triton('triton_poi_fused__softmax_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4f/c4fc6t37y6uwo5wssgfpkakc3dydrsjhgxebjkj7fmote6z7svdj.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_10, %view_14], -1), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_poi_fused_cat_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 2)
    x2 = xindex // 16
    x3 = xindex // 8
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + 2*(x0) + 8*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x3 + 32*((-4) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tr/ctre3ujjmahyho5bw56o2xvoshp6lzn64ll3ptwtwqklhyhg425r.py
# Topologically Sorted Source Nodes: [output, output_4], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   output => sum_3
#   output_4 => sum_9
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_16, [-2]), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_50, [-2]), kwargs = {})
triton_poi_fused_sum_16 = async_compile.triton('triton_poi_fused_sum_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x3 = xindex // 4
    x1 = ((xindex // 4) % 2)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*x3), xmask)
    tmp1 = tl.load(in_ptr1 + (2*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (2*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (2*x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (2*x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (4 + x0 + 8*x3), xmask)
    tmp17 = tl.load(in_ptr1 + (1 + 2*x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (1 + 2*x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (1 + 2*x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (1 + 2*x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr5 + (x0 + 8*x3), xmask)
    tmp35 = tl.load(in_ptr5 + (4 + x0 + 8*x3), xmask)
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp31 = tmp30 - tmp1
    tmp32 = tmp31 * tmp10
    tmp33 = tmp32 * tmp12
    tmp34 = tmp33 + tmp14
    tmp36 = tmp35 - tmp17
    tmp37 = tmp36 * tmp23
    tmp38 = tmp37 * tmp25
    tmp39 = tmp38 + tmp27
    tmp40 = tmp34 + tmp39
    tl.store(out_ptr0 + (x4), tmp29, xmask)
    tl.store(out_ptr1 + (x4), tmp40, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ef/cef3d7344cf3i6zbe7mwaf2xjqo524jl2twywqjvzfvbjidqd5hn.py
# Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_17 = async_compile.triton('triton_poi_fused_clone_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x3 + 8*x2 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + 8*y4), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/op/copzeeqgxe3xsppzhrs67a57swh32jctrbsy6z44vpb3gxpju6rv.py
# Topologically Sorted Source Nodes: [out_3, out_4], Original ATen: [aten.relu, aten.permute, aten.convolution]
# Source node to ATen node mapping:
#   out_3 => permute_40, relu_1
#   out_4 => convolution_3
# Graph fragment:
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%permute_38,), kwargs = {})
#   %permute_40 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_39, [0, 2, 1, 3]), kwargs = {})
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_40, %primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_permute_relu_18 = async_compile.triton('triton_poi_fused_convolution_permute_relu_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_permute_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_permute_relu_18(in_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 2)
    x3 = xindex // 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2 + 8*x1 + 32*x3), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr1 + (x0 + 4*x2 + 8*x1 + 32*x3), tmp2, xmask)
    tl.store(out_ptr2 + (x4), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cu/ccuzze7fne6glppotqcyzxloyagkg6ap65poexwsuakjwig45qq5.py
# Topologically Sorted Source Nodes: [out_5, out_6, out_7, out_13, out_14, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_13 => add_32, mul_58, mul_59, sub_19
#   out_14 => add_33
#   out_15 => relu_5
#   out_5 => add_15, mul_28, mul_29, sub_9
#   out_6 => add_16
#   out_7 => relu_2
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_67), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_69), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_71), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_73), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %primals_1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_67), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_69), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_71), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_73), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %primals_42), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
#   %le_46 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_5, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x1 = ((xindex // 16) % 4)
    x2 = xindex // 64
    x3 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x4), xmask)
    tmp22 = tl.load(in_ptr6 + (x4), xmask)
    tmp27 = tl.load(in_ptr7 + (x4), xmask)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = 0.0
    tmp21 = tmp19 <= tmp20
    tmp23 = tmp22 - tmp1
    tmp24 = tmp23 * tmp10
    tmp25 = tmp24 * tmp12
    tmp26 = tmp25 + tmp14
    tmp28 = tmp26 + tmp27
    tmp29 = triton_helpers.maximum(tmp18, tmp28)
    tl.store(out_ptr0 + (x3 + 128*x2), tmp19, xmask)
    tl.store(out_ptr1 + (x4), tmp21, xmask)
    tl.store(out_ptr2 + (x3 + 128*x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zb/czbh3nlulgeebe6mzkxa5kl3mldrwmci3ehcywicqcb6mnwytsdu.py
# Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   out_16 => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%cat_8, [None, None, %sub_21, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_21]), kwargs = {})
triton_poi_fused_reflection_pad2d_20 = async_compile.triton('triton_poi_fused_reflection_pad2d_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z2/cz273qrcyja4kwa2zz3lzmfce3gkrf2ihudqlsoiyuqryuxdvxma.py
# Topologically Sorted Source Nodes: [out_17, out_18, out_33, out_34, out_35], Original ATen: [aten.convolution, aten.relu, aten.add, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_17 => convolution_8
#   out_18 => relu_6
#   out_33 => convolution_13
#   out_34 => relu_11
#   out_35 => add_34
# Graph fragment:
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_43, %primals_44, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_11, %primals_51, %primals_52, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_6, %relu_11), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_11, 0), kwargs = {})
#   %le_43 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
triton_poi_fused_add_convolution_relu_threshold_backward_21 = async_compile.triton('triton_poi_fused_add_convolution_relu_threshold_backward_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_threshold_backward_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_threshold_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x3), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp7 = tmp5 + tmp6
    tmp8 = triton_helpers.maximum(tmp3, tmp7)
    tmp9 = tmp4 + tmp8
    tmp10 = 0.0
    tmp11 = tmp8 <= tmp10
    tmp12 = tmp4 <= tmp10
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
    tl.store(out_ptr2 + (x3), tmp12, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (2, ), (1, ))
    assert_size_stride(primals_4, (2, ), (1, ))
    assert_size_stride(primals_5, (2, ), (1, ))
    assert_size_stride(primals_6, (2, ), (1, ))
    assert_size_stride(primals_7, (4, 2, 1), (2, 1, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 7), (7, 1))
    assert_size_stride(primals_13, (16, ), (1, ))
    assert_size_stride(primals_14, (3, ), (1, ))
    assert_size_stride(primals_15, (3, ), (1, ))
    assert_size_stride(primals_16, (3, ), (1, ))
    assert_size_stride(primals_17, (3, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, 2, 1), (2, 1, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, ), (1, ))
    assert_size_stride(primals_27, (4, 7), (7, 1))
    assert_size_stride(primals_28, (16, ), (1, ))
    assert_size_stride(primals_29, (3, ), (1, ))
    assert_size_stride(primals_30, (3, ), (1, ))
    assert_size_stride(primals_31, (3, ), (1, ))
    assert_size_stride(primals_32, (3, ), (1, ))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_38, (4, ), (1, ))
    assert_size_stride(primals_39, (4, ), (1, ))
    assert_size_stride(primals_40, (4, ), (1, ))
    assert_size_stride(primals_41, (4, ), (1, ))
    assert_size_stride(primals_42, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_43, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_44, (4, ), (1, ))
    assert_size_stride(primals_45, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_46, (4, ), (1, ))
    assert_size_stride(primals_47, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_48, (4, ), (1, ))
    assert_size_stride(primals_49, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_50, (4, ), (1, ))
    assert_size_stride(primals_51, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_52, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf60 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_0.run(buf60, 4, grid=grid(4), stream=stream0)
        buf4 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [index_select], Original ATen: [aten.index_select]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_1.run(primals_13, primals_12, buf4, 64, grid=grid(64), stream=stream0)
        del primals_12
        buf16 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [index_select_1], Original ATen: [aten.index_select]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_1.run(primals_28, primals_27, buf16, 64, grid=grid(64), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 2, 4, 4), (32, 16, 4, 1))
        buf1 = empty_strided_cuda((4, 4, 2, 4), (32, 8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 16, 8, grid=grid(16, 8), stream=stream0)
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(primals_42, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 2, 4, 4), (32, 16, 4, 1))
        buf30 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        buf81 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_9, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_3.run(buf29, primals_3, primals_4, primals_5, primals_6, buf30, buf81, 128, grid=grid(128), stream=stream0)
        buf31 = empty_strided_cuda((4, 4, 2, 4), (32, 8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf30, buf31, 16, 8, grid=grid(16, 8), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(reinterpret_tensor(buf1, (16, 2, 4), (8, 4, 1), 0), primals_7, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf2, (16, 4, 4), (16, 4, 1))
        # Topologically Sorted Source Nodes: [conv1d_2], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(reinterpret_tensor(buf31, (16, 2, 4), (8, 4, 1), 0), primals_7, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf32, (16, 4, 4), (16, 4, 1))
        buf63 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_19], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(primals_42, buf63, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_45, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 4, 4, 4), (64, 16, 4, 1))
        buf65 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(primals_1, buf65, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_45, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 4, 4, 4), (64, 16, 4, 1))
        buf77 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf78 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_20, out_21, out_23, out_24], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_6.run(buf66, primals_46, buf64, buf77, buf78, 256, grid=grid(256), stream=stream0)
        buf67 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_25, out_26], Original ATen: [aten.cat, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_reflection_pad2d_7.run(buf64, primals_46, buf66, buf67, 512, grid=grid(512), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [out_27], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 4, 4, 4), (64, 16, 4, 1))
        buf76 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_27, out_28], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_8.run(buf68, primals_48, buf76, 256, grid=grid(256), stream=stream0)
        buf69 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_27, out_28, out_29], Original ATen: [aten.convolution, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_reflection_pad2d_relu_9.run(buf68, primals_48, buf69, 576, grid=grid(576), stream=stream0)
        del primals_48
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 4, 4, 4), (64, 16, 4, 1))
        buf75 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_30, out_31], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_8.run(buf70, primals_50, buf75, 256, grid=grid(256), stream=stream0)
        buf71 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_30, out_31, out_32], Original ATen: [aten.convolution, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_reflection_pad2d_relu_9.run(buf70, primals_50, buf71, 576, grid=grid(576), stream=stream0)
        del primals_50
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 4, 4, 4), (64, 16, 4, 1))
        buf3 = reinterpret_tensor(buf70, (16, 4, 4), (16, 4, 1), 0); del buf70  # reuse
        buf33 = reinterpret_tensor(buf68, (16, 4, 4), (16, 4, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_2], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf2, primals_8, primals_9, primals_10, primals_11, buf32, buf3, buf33, 256, grid=grid(256), stream=stream0)
        del primals_11
        buf5 = empty_strided_cuda((16, 3, 4, 4), (48, 16, 4, 1), torch.float32)
        buf34 = empty_strided_cuda((16, 3, 4, 4), (48, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stacked_similarity, stacked_similarity_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf3, buf4, buf33, buf5, buf34, 768, grid=grid(768), stream=stream0)
        buf6 = reinterpret_tensor(buf66, (16, 1, 4, 4), (16, 256, 4, 1), 0); del buf66  # reuse
        buf35 = reinterpret_tensor(buf64, (16, 1, 4, 4), (16, 256, 4, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [stacked_similarity_1, stacked_similarity_5], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_12.run(buf5, primals_14, primals_15, primals_16, primals_17, buf34, buf6, buf35, 256, grid=grid(256), stream=stream0)
        del primals_17
        buf7 = empty_strided_cuda((16, 1, 4, 4), (16, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [similarity], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_13.run(buf6, buf7, 256, grid=grid(256), stream=stream0)
        buf8 = reinterpret_tensor(buf6, (16, 1, 4, 4), (16, 16, 4, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [similarity], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_14.run(buf7, buf8, 256, grid=grid(256), stream=stream0)
        buf36 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [similarity_2], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_13.run(buf35, buf36, 256, grid=grid(256), stream=stream0)
        buf37 = reinterpret_tensor(buf35, (16, 1, 4, 4), (16, 16, 4, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [similarity_2], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_14.run(buf36, buf37, 256, grid=grid(256), stream=stream0)
        buf9 = reinterpret_tensor(buf30, (16, 4, 2), (8, 2, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [sv], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf3, (16, 4, 2), (16, 1, 4), 8), out=buf9)
        buf10 = empty_strided_cuda((4, 16, 2), (32, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sve], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (4, 16, 4), (4, 16, 1), 0), reinterpret_tensor(buf4, (4, 4, 2), (4, 1, 16), 32), out=buf10)
        buf11 = reinterpret_tensor(buf36, (16, 1, 2, 8), (16, 16, 8, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf9, buf10, buf11, 256, grid=grid(256), stream=stream0)
        buf38 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [sv_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf33, (16, 4, 2), (16, 1, 4), 8), out=buf38)
        buf39 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [sve_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (4, 16, 4), (4, 16, 1), 0), reinterpret_tensor(buf4, (4, 4, 2), (4, 1, 16), 32), out=buf39)
        buf40 = empty_strided_cuda((16, 1, 2, 8), (16, 16, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf38, buf39, buf40, 256, grid=grid(256), stream=stream0)
        buf12 = reinterpret_tensor(buf39, (4, 4, 2, 4), (32, 8, 4, 1), 0); del buf39  # reuse
        buf41 = reinterpret_tensor(buf38, (4, 4, 2, 4), (32, 8, 4, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [output, output_4], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_16.run(buf11, primals_18, primals_19, primals_20, primals_21, buf40, buf12, buf41, 128, grid=grid(128), stream=stream0)
        del primals_21
        buf13 = empty_strided_cuda((4, 4, 2, 4), (32, 8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_17.run(buf12, buf13, 16, 8, grid=grid(16, 8), stream=stream0)
        buf42 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_17.run(buf41, buf42, 16, 8, grid=grid(16, 8), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d_1], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(reinterpret_tensor(buf13, (16, 2, 4), (8, 4, 1), 0), primals_22, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf14, (16, 4, 4), (16, 4, 1))
        # Topologically Sorted Source Nodes: [conv1d_3], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(reinterpret_tensor(buf42, (16, 2, 4), (8, 4, 1), 0), primals_22, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf43, (16, 4, 4), (16, 4, 1))
        buf15 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        buf44 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [qkv_1, qkv_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf14, primals_23, primals_24, primals_25, primals_26, buf43, buf15, buf44, 256, grid=grid(256), stream=stream0)
        del primals_26
        buf17 = empty_strided_cuda((16, 3, 4, 4), (48, 16, 4, 1), torch.float32)
        buf45 = empty_strided_cuda((16, 3, 4, 4), (48, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stacked_similarity_2, stacked_similarity_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf15, buf16, buf44, buf17, buf45, 768, grid=grid(768), stream=stream0)
        buf18 = empty_strided_cuda((16, 1, 4, 4), (16, 256, 4, 1), torch.float32)
        buf46 = empty_strided_cuda((16, 1, 4, 4), (16, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stacked_similarity_3, stacked_similarity_7], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_12.run(buf17, primals_29, primals_30, primals_31, primals_32, buf45, buf18, buf46, 256, grid=grid(256), stream=stream0)
        del primals_32
        buf19 = empty_strided_cuda((16, 1, 4, 4), (16, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [similarity_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_13.run(buf18, buf19, 256, grid=grid(256), stream=stream0)
        buf20 = reinterpret_tensor(buf18, (16, 1, 4, 4), (16, 16, 4, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [similarity_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_14.run(buf19, buf20, 256, grid=grid(256), stream=stream0)
        buf47 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [similarity_3], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_13.run(buf46, buf47, 256, grid=grid(256), stream=stream0)
        buf48 = reinterpret_tensor(buf46, (16, 1, 4, 4), (16, 16, 4, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [similarity_3], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_14.run(buf47, buf48, 256, grid=grid(256), stream=stream0)
        buf21 = reinterpret_tensor(buf41, (16, 4, 2), (8, 2, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [sv_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf15, (16, 4, 2), (16, 1, 4), 8), out=buf21)
        buf22 = empty_strided_cuda((4, 16, 2), (32, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sve_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (4, 16, 4), (4, 16, 1), 0), reinterpret_tensor(buf16, (4, 4, 2), (4, 1, 16), 32), out=buf22)
        buf23 = reinterpret_tensor(buf47, (16, 1, 2, 8), (16, 16, 8, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf21, buf22, buf23, 256, grid=grid(256), stream=stream0)
        buf49 = reinterpret_tensor(buf22, (16, 4, 2), (8, 2, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [sv_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf44, (16, 4, 2), (16, 1, 4), 8), out=buf49)
        buf50 = reinterpret_tensor(buf21, (4, 16, 2), (32, 2, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [sve_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (4, 16, 4), (4, 16, 1), 0), reinterpret_tensor(buf16, (4, 4, 2), (4, 1, 16), 32), out=buf50)
        buf51 = empty_strided_cuda((16, 1, 2, 8), (16, 16, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf49, buf50, buf51, 256, grid=grid(256), stream=stream0)
        buf24 = reinterpret_tensor(buf50, (4, 4, 2, 4), (32, 8, 4, 1), 0); del buf50  # reuse
        buf52 = reinterpret_tensor(buf49, (4, 4, 2, 4), (32, 8, 4, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [output_2, output_6], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_16.run(buf23, primals_33, primals_34, primals_35, primals_36, buf51, buf24, buf52, 128, grid=grid(128), stream=stream0)
        del primals_36
        buf26 = empty_strided_cuda((4, 2, 4, 4), (32, 4, 8, 1), torch.float32)
        buf27 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3, out_4], Original ATen: [aten.relu, aten.permute, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_permute_relu_18.run(buf24, buf26, buf27, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 4, 4, 4), (64, 16, 4, 1))
        buf54 = reinterpret_tensor(buf27, (4, 2, 4, 4), (32, 4, 8, 1), 0); del buf27  # reuse
        buf55 = reinterpret_tensor(buf24, (4, 2, 4, 4), (32, 16, 4, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_11, out_12], Original ATen: [aten.relu, aten.permute, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_permute_relu_18.run(buf52, buf54, buf55, 128, grid=grid(128), stream=stream0)
        del buf52
        # Topologically Sorted Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 4, 4, 4), (64, 16, 4, 1))
        del buf55
        buf59 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        buf57 = reinterpret_tensor(buf59, (4, 4, 4, 4), (128, 16, 4, 1), 64)  # alias
        buf80 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf58 = reinterpret_tensor(buf59, (4, 4, 4, 4), (128, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [out_5, out_6, out_7, out_13, out_14, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_19.run(buf56, primals_38, primals_39, primals_40, primals_41, primals_42, buf28, primals_1, buf57, buf80, buf58, 256, grid=grid(256), stream=stream0)
        buf61 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_20.run(buf59, buf61, 512, grid=grid(512), stream=stream0)
        del buf57
        del buf58
        del buf59
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 4, 4, 4), (64, 16, 4, 1))
        buf73 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf74 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf79 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_17, out_18, out_33, out_34, out_35], Original ATen: [aten.convolution, aten.relu, aten.add, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_threshold_backward_21.run(buf62, primals_44, buf72, primals_52, buf73, buf74, buf79, 256, grid=grid(256), stream=stream0)
        del buf62
        del buf72
        del primals_44
        del primals_52
    return (buf73, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_45, primals_47, primals_49, primals_51, buf0, reinterpret_tensor(buf1, (16, 2, 4), (8, 4, 1), 0), buf2, reinterpret_tensor(buf3, (16, 1, 4, 1, 1), (16, 16, 1, 1, 4), 0), reinterpret_tensor(buf4, (1, 1, 4, 4, 1), (1, 1, 4, 1, 16), 0), reinterpret_tensor(buf3, (16, 1, 1, 4, 1), (16, 16, 4, 1, 1), 4), reinterpret_tensor(buf4, (1, 1, 4, 4, 1), (1, 1, 4, 1, 16), 16), buf5, buf8, buf11, reinterpret_tensor(buf13, (16, 2, 4), (8, 4, 1), 0), buf14, reinterpret_tensor(buf15, (16, 1, 4, 1, 1), (16, 16, 1, 1, 4), 0), reinterpret_tensor(buf16, (1, 1, 4, 4, 1), (1, 1, 4, 1, 16), 0), reinterpret_tensor(buf15, (16, 1, 1, 4, 1), (16, 16, 4, 1, 1), 4), reinterpret_tensor(buf16, (1, 1, 4, 4, 1), (1, 1, 4, 1, 16), 16), buf17, buf20, buf23, buf26, buf28, buf29, reinterpret_tensor(buf31, (16, 2, 4), (8, 4, 1), 0), buf32, reinterpret_tensor(buf33, (16, 1, 4, 1, 1), (16, 16, 1, 1, 4), 0), reinterpret_tensor(buf33, (16, 1, 1, 4, 1), (16, 16, 4, 1, 1), 4), buf34, buf37, buf40, reinterpret_tensor(buf42, (16, 2, 4), (8, 4, 1), 0), buf43, reinterpret_tensor(buf44, (16, 1, 4, 1, 1), (16, 16, 1, 1, 4), 0), reinterpret_tensor(buf44, (16, 1, 1, 4, 1), (16, 16, 4, 1, 1), 4), buf45, buf48, buf51, buf54, buf56, buf60, buf61, buf63, buf65, buf67, buf69, buf71, buf74, buf75, buf76, buf77, buf78, buf79, buf80, reinterpret_tensor(buf16, (4, 2, 4), (4, 16, 1), 32), reinterpret_tensor(buf44, (16, 2, 4), (16, 4, 1), 8), reinterpret_tensor(buf4, (4, 2, 4), (4, 16, 1), 32), reinterpret_tensor(buf33, (16, 2, 4), (16, 4, 1), 8), buf81, reinterpret_tensor(buf15, (16, 2, 4), (16, 4, 1), 8), reinterpret_tensor(buf3, (16, 2, 4), (16, 4, 1), 8), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 7), (7, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_14 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, 7), (7, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_29 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
