# AOT ID: ['14_forward']
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


# kernel path: inductor_cache/4e/c4e4jwedft3woxyvdin4ge2zn6a4eri3gakgi23xo6cjj5anrrku.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %le_11 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_convolution_relu_threshold_backward_0 = async_compile.triton('triton_poi_fused_convolution_relu_threshold_backward_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/l3/cl3s7srt5hzcl234eiguoyirb7olvcwmtcovujlhew4cz6phudd2.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_3, %relu], 1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 96)
    x0 = (xindex % 4096)
    x2 = xindex // 393216
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 131072*x2), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-64) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/ub/cubjicew52w27i5a4oxlr2t2cyh4kjcrpqginmh4hjvgv3l7orlz.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_3, %relu, %relu_1], 1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 128)
    x0 = (xindex % 4096)
    x2 = xindex // 524288
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 131072*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-64) + x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp9, tmp14, tmp15)
    tmp17 = tmp0 >= tmp7
    tmp18 = tl.full([1], 128, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4096*((-96) + x1) + 131072*x2), tmp17, other=0.0)
    tmp21 = tl.load(in_ptr4 + ((-96) + x1), tmp17, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full([1], 0, tl.int32)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp17, tmp24, tmp25)
    tmp27 = tl.where(tmp9, tmp16, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/el/celser5p6fy75geg5hvuaycthcnbproirchghsdetqtmvqvtd226.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_3, %relu, %relu_1, %relu_2], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 160)
    x0 = (xindex % 4096)
    x2 = xindex // 655360
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 131072*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-64) + x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp9, tmp14, tmp15)
    tmp17 = tmp0 >= tmp7
    tmp18 = tl.full([1], 128, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 4096*((-96) + x1) + 131072*x2), tmp20, other=0.0)
    tmp22 = tl.load(in_ptr4 + ((-96) + x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp20, tmp25, tmp26)
    tmp28 = tmp0 >= tmp18
    tmp29 = tl.full([1], 160, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr5 + (x0 + 4096*((-128) + x1) + 131072*x2), tmp28, other=0.0)
    tmp32 = tl.load(in_ptr6 + ((-128) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp28, tmp35, tmp36)
    tmp38 = tl.where(tmp20, tmp27, tmp37)
    tmp39 = tl.where(tmp9, tmp16, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tl.store(out_ptr0 + (x3), tmp40, None)
''', device_str='cuda')


# kernel path: inductor_cache/6w/c6wex7ladwvxuhykizxywjhr3b6xnx33qnotnwkc2ppkmrsxnsba.py
# Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_3, %relu, %relu_1, %relu_2, %relu_3], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 192)
    x0 = (xindex % 4096)
    x2 = xindex // 786432
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 131072*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-64) + x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp9, tmp14, tmp15)
    tmp17 = tmp0 >= tmp7
    tmp18 = tl.full([1], 128, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 4096*((-96) + x1) + 131072*x2), tmp20, other=0.0)
    tmp22 = tl.load(in_ptr4 + ((-96) + x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp20, tmp25, tmp26)
    tmp28 = tmp0 >= tmp18
    tmp29 = tl.full([1], 160, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tl.load(in_ptr5 + (x0 + 4096*((-128) + x1) + 131072*x2), tmp31, other=0.0)
    tmp33 = tl.load(in_ptr6 + ((-128) + x1), tmp31, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full([1], 0, tl.int32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp31, tmp36, tmp37)
    tmp39 = tmp0 >= tmp29
    tmp40 = tl.full([1], 192, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr7 + (x0 + 4096*((-160) + x1) + 131072*x2), tmp39, other=0.0)
    tmp43 = tl.load(in_ptr8 + ((-160) + x1), tmp39, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp39, tmp46, tmp47)
    tmp49 = tl.where(tmp31, tmp38, tmp48)
    tmp50 = tl.where(tmp20, tmp27, tmp49)
    tmp51 = tl.where(tmp9, tmp16, tmp50)
    tmp52 = tl.where(tmp4, tmp5, tmp51)
    tl.store(out_ptr0 + (x3), tmp52, None)
''', device_str='cuda')


# kernel path: inductor_cache/en/cenpb2aozlbhdiqolcowivzyt5aesli7rkanpkr2ibb7jmrw27yd.py
# Topologically Sorted Source Nodes: [x5, mul_, out], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_ => mul
#   out => add
#   x5 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %primals_10, %primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_4, 0.2), kwargs = {})
#   %add : [num_users=7] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %primals_3), kwargs = {})
triton_poi_fused_add_convolution_mul_5 = async_compile.triton('triton_poi_fused_add_convolution_mul_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/s6/cs6qxvxg7jsevywcf6hqrjv42v3uxib3b2pjvttnw7n3j7fnfjnb.py
# Topologically Sorted Source Nodes: [x5_2, mul__2, out_2, mul__3, add_3], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_3
#   mul__2 => mul_2
#   mul__3 => mul_3
#   out_2 => add_2
#   x5_2 => convolution_14
# Graph fragment:
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %primals_30, %primals_31, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_14, 0.2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %add_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_3), kwargs = {})
triton_poi_fused_add_convolution_mul_6 = async_compile.triton('triton_poi_fused_add_convolution_mul_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp3
    tmp9 = tmp7 + tmp8
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31 = args
    args.clear()
    assert_size_stride(primals_1, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (4, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(primals_4, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_31, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf41 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf0, primals_2, buf41, 524288, grid=grid(524288), stream=stream0)
        buf1 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(primals_3, buf0, primals_2, buf1, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf40 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf2, primals_5, buf40, 524288, grid=grid(524288), stream=stream0)
        buf3 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(primals_3, buf0, primals_2, buf2, primals_5, buf3, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf39 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf4, primals_7, buf39, 524288, grid=grid(524288), stream=stream0)
        buf5 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(primals_3, buf0, primals_2, buf2, primals_5, buf4, primals_7, buf5, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf38 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf6, primals_9, buf38, 524288, grid=grid(524288), stream=stream0)
        buf7 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(primals_3, buf0, primals_2, buf2, primals_5, buf4, primals_7, buf6, primals_9, buf7, 3145728, grid=grid(3145728), stream=stream0)
        del buf0
        del buf2
        del buf4
        del buf6
        del primals_2
        del primals_5
        del primals_7
        del primals_9
        # Topologically Sorted Source Nodes: [x5], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x5, mul_, out], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_5.run(buf9, primals_11, primals_3, 1048576, grid=grid(1048576), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf37 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf10, primals_13, buf37, 524288, grid=grid(524288), stream=stream0)
        buf11 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf9, buf10, primals_13, buf11, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf36 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf12, primals_15, buf36, 524288, grid=grid(524288), stream=stream0)
        buf13 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf9, buf10, primals_13, buf12, primals_15, buf13, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf35 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf14, primals_17, buf35, 524288, grid=grid(524288), stream=stream0)
        buf15 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf9, buf10, primals_13, buf12, primals_15, buf14, primals_17, buf15, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf34 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf16, primals_19, buf34, 524288, grid=grid(524288), stream=stream0)
        buf17 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf9, buf10, primals_13, buf12, primals_15, buf14, primals_17, buf16, primals_19, buf17, 3145728, grid=grid(3145728), stream=stream0)
        del buf10
        del buf12
        del buf14
        del buf16
        del primals_13
        del primals_15
        del primals_17
        del primals_19
        # Topologically Sorted Source Nodes: [x5_1], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x5_1, mul__1, out_1], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_5.run(buf19, primals_21, buf9, 1048576, grid=grid(1048576), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf33 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf20, primals_23, buf33, 524288, grid=grid(524288), stream=stream0)
        buf21 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf19, buf20, primals_23, buf21, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf32 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf22, primals_25, buf32, 524288, grid=grid(524288), stream=stream0)
        buf23 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf19, buf20, primals_23, buf22, primals_25, buf23, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf31 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf24, primals_27, buf31, 524288, grid=grid(524288), stream=stream0)
        buf25 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf19, buf20, primals_23, buf22, primals_25, buf24, primals_27, buf25, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf30 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_0.run(buf26, primals_29, buf30, 524288, grid=grid(524288), stream=stream0)
        buf27 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf19, buf20, primals_23, buf22, primals_25, buf24, primals_27, buf26, primals_29, buf27, 3145728, grid=grid(3145728), stream=stream0)
        del buf20
        del buf22
        del buf24
        del buf26
        del primals_23
        del primals_25
        del primals_27
        del primals_29
        # Topologically Sorted Source Nodes: [x5_2], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x5_2, mul__2, out_2, mul__3, add_3], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf29, primals_31, buf19, primals_3, 1048576, grid=grid(1048576), stream=stream0)
        del primals_31
    return (buf29, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf27, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
