# AOT ID: ['26_forward']
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


# kernel path: inductor_cache/zp/czp35rtqpmtwk6q4hww5ukuytylpkbmm4232sdx4ja2dsxqlxtvb.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %convolution : [num_users=9] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k2/ck2ydru6abbclpfbioyatf4z2zfvem6gfsxw7mkfzog2ttpxwtc5.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where], 1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 36)
    x0 = (xindex % 16)
    x2 = xindex // 576
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 16*((-4) + x1) + 512*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-4) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.2
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g6/cg66gm3bfjxzyqj6fifzyncmkvt237npsat7mdpe4x43uvwltrho.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   input_2 => convolution_1
#   input_3 => gt, mul, where
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_4, %primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.2), kwargs = {})
#   %where : [num_users=5] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_1, %mul), kwargs = {})
#   %gt_97 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = tmp7 > tmp3
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zo/czo7yz73xtps5lhwdbpx3mqa25lyh7orbrb4daxsu2paipuz2sfn.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where, %where_1], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 68)
    x0 = (xindex % 16)
    x2 = xindex // 1088
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-4) + x1) + 512*x2), tmp9 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-4) + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.2
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 68, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr3 + (x0 + 16*((-36) + x1) + 512*x2), tmp20 & xmask, other=0.0)
    tmp24 = tl.load(in_ptr4 + ((-36) + x1), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp28 = 0.2
    tmp29 = tmp25 * tmp28
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp20, tmp30, tmp31)
    tmp33 = tl.where(tmp9, tmp19, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tl.store(out_ptr0 + (x3), tmp34, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/le/clej2mwuzcs7aeaojr2zmessiukxjdujtengyzzd6kv4caiomwmc.py
# Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where, %where_1, %where_2], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 100)
    x0 = (xindex % 16)
    x2 = xindex // 1600
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-4) + x1) + 512*x2), tmp9 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-4) + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.2
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 68, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr3 + (x0 + 16*((-36) + x1) + 512*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.load(in_ptr4 + ((-36) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = 0.0
    tmp28 = tmp26 > tmp27
    tmp29 = 0.2
    tmp30 = tmp26 * tmp29
    tmp31 = tl.where(tmp28, tmp26, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tmp0 >= tmp21
    tmp35 = tl.full([1], 100, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr5 + (x0 + 16*((-68) + x1) + 512*x2), tmp34 & xmask, other=0.0)
    tmp38 = tl.load(in_ptr6 + ((-68) + x1), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = 0.0
    tmp41 = tmp39 > tmp40
    tmp42 = 0.2
    tmp43 = tmp39 * tmp42
    tmp44 = tl.where(tmp41, tmp39, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp34, tmp44, tmp45)
    tmp47 = tl.where(tmp23, tmp33, tmp46)
    tmp48 = tl.where(tmp9, tmp19, tmp47)
    tmp49 = tl.where(tmp4, tmp5, tmp48)
    tl.store(out_ptr0 + (x3), tmp49, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s7/cs7aqohwwya7m3swdq2ee23ut3odmeawjb7n2p6ecvl36xy2o33y.py
# Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_4 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where, %where_1, %where_2, %where_3], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 132)
    x0 = (xindex % 16)
    x2 = xindex // 2112
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-4) + x1) + 512*x2), tmp9 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-4) + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.2
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 68, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr3 + (x0 + 16*((-36) + x1) + 512*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.load(in_ptr4 + ((-36) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = 0.0
    tmp28 = tmp26 > tmp27
    tmp29 = 0.2
    tmp30 = tmp26 * tmp29
    tmp31 = tl.where(tmp28, tmp26, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tmp0 >= tmp21
    tmp35 = tl.full([1], 100, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tl.load(in_ptr5 + (x0 + 16*((-68) + x1) + 512*x2), tmp37 & xmask, other=0.0)
    tmp39 = tl.load(in_ptr6 + ((-68) + x1), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 + tmp39
    tmp41 = 0.0
    tmp42 = tmp40 > tmp41
    tmp43 = 0.2
    tmp44 = tmp40 * tmp43
    tmp45 = tl.where(tmp42, tmp40, tmp44)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp37, tmp45, tmp46)
    tmp48 = tmp0 >= tmp35
    tmp49 = tl.full([1], 132, tl.int64)
    tmp50 = tmp0 < tmp49
    tmp51 = tl.load(in_ptr7 + (x0 + 16*((-100) + x1) + 512*x2), tmp48 & xmask, other=0.0)
    tmp52 = tl.load(in_ptr8 + ((-100) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = tmp51 + tmp52
    tmp54 = 0.0
    tmp55 = tmp53 > tmp54
    tmp56 = 0.2
    tmp57 = tmp53 * tmp56
    tmp58 = tl.where(tmp55, tmp53, tmp57)
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp48, tmp58, tmp59)
    tmp61 = tl.where(tmp37, tmp47, tmp60)
    tmp62 = tl.where(tmp23, tmp33, tmp61)
    tmp63 = tl.where(tmp9, tmp19, tmp62)
    tmp64 = tl.where(tmp4, tmp5, tmp63)
    tl.store(out_ptr0 + (x3), tmp64, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oh/cohdpfrcdrk3mjyxfcuoe5brcuyx44f2x52rbljg7f2lsjtfssc7.py
# Topologically Sorted Source Nodes: [input_10, mul, out], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_10 => convolution_5
#   mul => mul_4
#   out => add
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %primals_12, %primals_13, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_5, 0.2), kwargs = {})
#   %add : [num_users=7] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %convolution), kwargs = {})
triton_poi_fused_add_convolution_mul_6 = async_compile.triton('triton_poi_fused_add_convolution_mul_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_6(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/px/cpxczfmn64fjfx425jklbs755ct4kxiucbqlcgdia4boubxtzxau.py
# Topologically Sorted Source Nodes: [input_28, mul_2, out_2, mul_3, input_29], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_28 => convolution_15
#   input_29 => add_3
#   mul_2 => mul_14
#   mul_3 => mul_15
#   out_2 => add_2
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %primals_32, %primals_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, 0.2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %add_1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.2), kwargs = {})
#   %add_3 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %convolution), kwargs = {})
triton_poi_fused_add_convolution_mul_7 = async_compile.triton('triton_poi_fused_add_convolution_mul_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp8 = tl.load(in_ptr2 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp3
    tmp9 = tmp7 + tmp8
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r5/cr53c22yb47o5mk7ukgxkbfjq2grthljlgzypugldqesuxknl75n.py
# Topologically Sorted Source Nodes: [input_114, x], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_114 => convolution_61
#   x => add_16
# Graph fragment:
#   %convolution_61 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_15, %primals_124, %primals_125, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_61, %convolution), kwargs = {})
triton_poi_fused_add_convolution_8 = async_compile.triton('triton_poi_fused_add_convolution_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tk/ctk3htdhgxeg574p7az4zpxaporab3fnb2gujfmgbcm2lophzzmz.py
# Topologically Sorted Source Nodes: [input_115, input_116], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_115 => convolution_62
#   input_116 => gt_48, mul_64, where_48
# Graph fragment:
#   %convolution_62 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_16, %primals_126, %primals_127, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_48 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_62, 0), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_62, 0.2), kwargs = {})
#   %where_48 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_48, %convolution_62, %mul_64), kwargs = {})
triton_poi_fused_convolution_leaky_relu_9 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_31, (32, ), (1, ))
    assert_size_stride(primals_32, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_37, (32, ), (1, ))
    assert_size_stride(primals_38, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_39, (32, ), (1, ))
    assert_size_stride(primals_40, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_41, (32, ), (1, ))
    assert_size_stride(primals_42, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_43, (4, ), (1, ))
    assert_size_stride(primals_44, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_47, (32, ), (1, ))
    assert_size_stride(primals_48, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_49, (32, ), (1, ))
    assert_size_stride(primals_50, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_51, (32, ), (1, ))
    assert_size_stride(primals_52, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_53, (4, ), (1, ))
    assert_size_stride(primals_54, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_55, (32, ), (1, ))
    assert_size_stride(primals_56, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_57, (32, ), (1, ))
    assert_size_stride(primals_58, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_59, (32, ), (1, ))
    assert_size_stride(primals_60, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_63, (4, ), (1, ))
    assert_size_stride(primals_64, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_65, (32, ), (1, ))
    assert_size_stride(primals_66, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_67, (32, ), (1, ))
    assert_size_stride(primals_68, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_69, (32, ), (1, ))
    assert_size_stride(primals_70, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_71, (32, ), (1, ))
    assert_size_stride(primals_72, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_73, (4, ), (1, ))
    assert_size_stride(primals_74, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_75, (32, ), (1, ))
    assert_size_stride(primals_76, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_77, (32, ), (1, ))
    assert_size_stride(primals_78, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_79, (32, ), (1, ))
    assert_size_stride(primals_80, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_81, (32, ), (1, ))
    assert_size_stride(primals_82, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_83, (4, ), (1, ))
    assert_size_stride(primals_84, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_85, (32, ), (1, ))
    assert_size_stride(primals_86, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_87, (32, ), (1, ))
    assert_size_stride(primals_88, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_89, (32, ), (1, ))
    assert_size_stride(primals_90, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_91, (32, ), (1, ))
    assert_size_stride(primals_92, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_93, (4, ), (1, ))
    assert_size_stride(primals_94, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_97, (32, ), (1, ))
    assert_size_stride(primals_98, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_101, (32, ), (1, ))
    assert_size_stride(primals_102, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_103, (4, ), (1, ))
    assert_size_stride(primals_104, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_105, (32, ), (1, ))
    assert_size_stride(primals_106, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_107, (32, ), (1, ))
    assert_size_stride(primals_108, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_109, (32, ), (1, ))
    assert_size_stride(primals_110, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_113, (4, ), (1, ))
    assert_size_stride(primals_114, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_115, (32, ), (1, ))
    assert_size_stride(primals_116, (32, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_117, (32, ), (1, ))
    assert_size_stride(primals_118, (32, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_119, (32, ), (1, ))
    assert_size_stride(primals_120, (32, 100, 3, 3), (900, 9, 3, 1))
    assert_size_stride(primals_121, (32, ), (1, ))
    assert_size_stride(primals_122, (4, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_123, (4, ), (1, ))
    assert_size_stride(primals_124, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_125, (4, ), (1, ))
    assert_size_stride(primals_126, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_127, (4, ), (1, ))
    assert_size_stride(primals_128, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_129, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 4, 4), (64, 16, 4, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf1, primals_2, 256, grid=grid(256), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 4, 4), (512, 16, 4, 1))
        buf3 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf1, buf2, primals_5, buf3, 2304, grid=grid(2304), stream=stream0)
        buf175 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf2, primals_5, buf175, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 32, 4, 4), (512, 16, 4, 1))
        buf5 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf1, buf2, primals_5, buf4, primals_7, buf5, 4352, grid=grid(4352), stream=stream0)
        buf174 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf4, primals_7, buf174, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 32, 4, 4), (512, 16, 4, 1))
        buf7 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf1, buf2, primals_5, buf4, primals_7, buf6, primals_9, buf7, 6400, grid=grid(6400), stream=stream0)
        buf173 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf6, primals_9, buf173, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 32, 4, 4), (512, 16, 4, 1))
        buf9 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf1, buf2, primals_5, buf4, primals_7, buf6, primals_9, buf8, primals_11, buf9, 8448, grid=grid(8448), stream=stream0)
        del buf2
        del buf4
        del buf6
        del primals_5
        del primals_7
        del primals_9
        buf172 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf8, primals_11, buf172, 2048, grid=grid(2048), stream=stream0)
        del buf8
        del primals_11
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 4, 4, 4), (64, 16, 4, 1))
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_10, mul, out], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf11, primals_13, buf1, 256, grid=grid(256), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 32, 4, 4), (512, 16, 4, 1))
        buf13 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf11, buf12, primals_15, buf13, 2304, grid=grid(2304), stream=stream0)
        buf171 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf12, primals_15, buf171, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 32, 4, 4), (512, 16, 4, 1))
        buf15 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf11, buf12, primals_15, buf14, primals_17, buf15, 4352, grid=grid(4352), stream=stream0)
        buf170 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf14, primals_17, buf170, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 32, 4, 4), (512, 16, 4, 1))
        buf17 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf11, buf12, primals_15, buf14, primals_17, buf16, primals_19, buf17, 6400, grid=grid(6400), stream=stream0)
        buf169 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf16, primals_19, buf169, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 4, 4), (512, 16, 4, 1))
        buf19 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf11, buf12, primals_15, buf14, primals_17, buf16, primals_19, buf18, primals_21, buf19, 8448, grid=grid(8448), stream=stream0)
        del buf12
        del buf14
        del buf16
        del primals_15
        del primals_17
        del primals_19
        buf168 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf18, primals_21, buf168, 2048, grid=grid(2048), stream=stream0)
        del buf18
        del primals_21
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 4, 4, 4), (64, 16, 4, 1))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_19, mul_1, out_1], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf21, primals_23, buf11, 256, grid=grid(256), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 32, 4, 4), (512, 16, 4, 1))
        buf23 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf21, buf22, primals_25, buf23, 2304, grid=grid(2304), stream=stream0)
        buf167 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf22, primals_25, buf167, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 32, 4, 4), (512, 16, 4, 1))
        buf25 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf21, buf22, primals_25, buf24, primals_27, buf25, 4352, grid=grid(4352), stream=stream0)
        buf166 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf24, primals_27, buf166, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 32, 4, 4), (512, 16, 4, 1))
        buf27 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf21, buf22, primals_25, buf24, primals_27, buf26, primals_29, buf27, 6400, grid=grid(6400), stream=stream0)
        buf165 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf26, primals_29, buf165, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 32, 4, 4), (512, 16, 4, 1))
        buf29 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf21, buf22, primals_25, buf24, primals_27, buf26, primals_29, buf28, primals_31, buf29, 8448, grid=grid(8448), stream=stream0)
        del buf22
        del buf24
        del buf26
        del primals_25
        del primals_27
        del primals_29
        buf164 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf28, primals_31, buf164, 2048, grid=grid(2048), stream=stream0)
        del buf28
        del primals_31
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 4, 4, 4), (64, 16, 4, 1))
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [input_28, mul_2, out_2, mul_3, input_29], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf31, primals_33, buf21, buf1, 256, grid=grid(256), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 32, 4, 4), (512, 16, 4, 1))
        buf33 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf31, buf32, primals_35, buf33, 2304, grid=grid(2304), stream=stream0)
        buf163 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_30, input_31], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf32, primals_35, buf163, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 32, 4, 4), (512, 16, 4, 1))
        buf35 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf31, buf32, primals_35, buf34, primals_37, buf35, 4352, grid=grid(4352), stream=stream0)
        buf162 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf34, primals_37, buf162, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 32, 4, 4), (512, 16, 4, 1))
        buf37 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_18], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf31, buf32, primals_35, buf34, primals_37, buf36, primals_39, buf37, 6400, grid=grid(6400), stream=stream0)
        buf161 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf36, primals_39, buf161, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 32, 4, 4), (512, 16, 4, 1))
        buf39 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_19], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf31, buf32, primals_35, buf34, primals_37, buf36, primals_39, buf38, primals_41, buf39, 8448, grid=grid(8448), stream=stream0)
        del buf32
        del buf34
        del buf36
        del primals_35
        del primals_37
        del primals_39
        buf160 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf38, primals_41, buf160, 2048, grid=grid(2048), stream=stream0)
        del buf38
        del primals_41
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 4, 4, 4), (64, 16, 4, 1))
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_38, mul_4, out_3], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf41, primals_43, buf31, 256, grid=grid(256), stream=stream0)
        del primals_43
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 32, 4, 4), (512, 16, 4, 1))
        buf43 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_21], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf41, buf42, primals_45, buf43, 2304, grid=grid(2304), stream=stream0)
        buf159 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_39, input_40], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf42, primals_45, buf159, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 32, 4, 4), (512, 16, 4, 1))
        buf45 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_22], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf41, buf42, primals_45, buf44, primals_47, buf45, 4352, grid=grid(4352), stream=stream0)
        buf158 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_41, input_42], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf44, primals_47, buf158, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 32, 4, 4), (512, 16, 4, 1))
        buf47 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_23], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf41, buf42, primals_45, buf44, primals_47, buf46, primals_49, buf47, 6400, grid=grid(6400), stream=stream0)
        buf157 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf46, primals_49, buf157, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 32, 4, 4), (512, 16, 4, 1))
        buf49 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_24], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf41, buf42, primals_45, buf44, primals_47, buf46, primals_49, buf48, primals_51, buf49, 8448, grid=grid(8448), stream=stream0)
        del buf42
        del buf44
        del buf46
        del primals_45
        del primals_47
        del primals_49
        buf156 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf48, primals_51, buf156, 2048, grid=grid(2048), stream=stream0)
        del buf48
        del primals_51
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 4, 4, 4), (64, 16, 4, 1))
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_47, mul_5, out_4], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf51, primals_53, buf41, 256, grid=grid(256), stream=stream0)
        del primals_53
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 32, 4, 4), (512, 16, 4, 1))
        buf53 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_26], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf51, buf52, primals_55, buf53, 2304, grid=grid(2304), stream=stream0)
        buf155 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf52, primals_55, buf155, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 32, 4, 4), (512, 16, 4, 1))
        buf55 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_27], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf51, buf52, primals_55, buf54, primals_57, buf55, 4352, grid=grid(4352), stream=stream0)
        buf154 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_50, input_51], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf54, primals_57, buf154, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 32, 4, 4), (512, 16, 4, 1))
        buf57 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf51, buf52, primals_55, buf54, primals_57, buf56, primals_59, buf57, 6400, grid=grid(6400), stream=stream0)
        buf153 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf56, primals_59, buf153, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 32, 4, 4), (512, 16, 4, 1))
        buf59 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_29], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf51, buf52, primals_55, buf54, primals_57, buf56, primals_59, buf58, primals_61, buf59, 8448, grid=grid(8448), stream=stream0)
        del buf52
        del buf54
        del buf56
        del primals_55
        del primals_57
        del primals_59
        buf152 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf58, primals_61, buf152, 2048, grid=grid(2048), stream=stream0)
        del buf58
        del primals_61
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 4, 4, 4), (64, 16, 4, 1))
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_56, mul_6, out_5, mul_7, input_57], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf61, primals_63, buf51, buf31, 256, grid=grid(256), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 32, 4, 4), (512, 16, 4, 1))
        buf63 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_31], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf61, buf62, primals_65, buf63, 2304, grid=grid(2304), stream=stream0)
        buf151 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_58, input_59], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf62, primals_65, buf151, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 32, 4, 4), (512, 16, 4, 1))
        buf65 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_32], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf61, buf62, primals_65, buf64, primals_67, buf65, 4352, grid=grid(4352), stream=stream0)
        buf150 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf64, primals_67, buf150, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 32, 4, 4), (512, 16, 4, 1))
        buf67 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_33], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf61, buf62, primals_65, buf64, primals_67, buf66, primals_69, buf67, 6400, grid=grid(6400), stream=stream0)
        buf149 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_62, input_63], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf66, primals_69, buf149, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 32, 4, 4), (512, 16, 4, 1))
        buf69 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf61, buf62, primals_65, buf64, primals_67, buf66, primals_69, buf68, primals_71, buf69, 8448, grid=grid(8448), stream=stream0)
        del buf62
        del buf64
        del buf66
        del primals_65
        del primals_67
        del primals_69
        buf148 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_64, input_65], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf68, primals_71, buf148, 2048, grid=grid(2048), stream=stream0)
        del buf68
        del primals_71
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 4, 4, 4), (64, 16, 4, 1))
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_66, mul_8, out_6], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf71, primals_73, buf61, 256, grid=grid(256), stream=stream0)
        del primals_73
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 32, 4, 4), (512, 16, 4, 1))
        buf73 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_36], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf71, buf72, primals_75, buf73, 2304, grid=grid(2304), stream=stream0)
        buf147 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_67, input_68], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf72, primals_75, buf147, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 32, 4, 4), (512, 16, 4, 1))
        buf75 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf71, buf72, primals_75, buf74, primals_77, buf75, 4352, grid=grid(4352), stream=stream0)
        buf146 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_69, input_70], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf74, primals_77, buf146, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 32, 4, 4), (512, 16, 4, 1))
        buf77 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf71, buf72, primals_75, buf74, primals_77, buf76, primals_79, buf77, 6400, grid=grid(6400), stream=stream0)
        buf145 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_71, input_72], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf76, primals_79, buf145, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 32, 4, 4), (512, 16, 4, 1))
        buf79 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_39], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf71, buf72, primals_75, buf74, primals_77, buf76, primals_79, buf78, primals_81, buf79, 8448, grid=grid(8448), stream=stream0)
        del buf72
        del buf74
        del buf76
        del primals_75
        del primals_77
        del primals_79
        buf144 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_73, input_74], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf78, primals_81, buf144, 2048, grid=grid(2048), stream=stream0)
        del buf78
        del primals_81
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 4, 4, 4), (64, 16, 4, 1))
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [input_75, mul_9, out_7], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf81, primals_83, buf71, 256, grid=grid(256), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 32, 4, 4), (512, 16, 4, 1))
        buf83 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf81, buf82, primals_85, buf83, 2304, grid=grid(2304), stream=stream0)
        buf143 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_76, input_77], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf82, primals_85, buf143, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 32, 4, 4), (512, 16, 4, 1))
        buf85 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_42], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf81, buf82, primals_85, buf84, primals_87, buf85, 4352, grid=grid(4352), stream=stream0)
        buf142 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_78, input_79], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf84, primals_87, buf142, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 32, 4, 4), (512, 16, 4, 1))
        buf87 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_43], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf81, buf82, primals_85, buf84, primals_87, buf86, primals_89, buf87, 6400, grid=grid(6400), stream=stream0)
        buf141 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf86, primals_89, buf141, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 32, 4, 4), (512, 16, 4, 1))
        buf89 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf81, buf82, primals_85, buf84, primals_87, buf86, primals_89, buf88, primals_91, buf89, 8448, grid=grid(8448), stream=stream0)
        del buf82
        del buf84
        del buf86
        del primals_85
        del primals_87
        del primals_89
        buf140 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_82, input_83], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf88, primals_91, buf140, 2048, grid=grid(2048), stream=stream0)
        del buf88
        del primals_91
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 4, 4, 4), (64, 16, 4, 1))
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [input_84, mul_10, out_8, mul_11, input_85], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf91, primals_93, buf81, buf61, 256, grid=grid(256), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 32, 4, 4), (512, 16, 4, 1))
        buf93 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_46], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf91, buf92, primals_95, buf93, 2304, grid=grid(2304), stream=stream0)
        buf139 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_86, input_87], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf92, primals_95, buf139, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 32, 4, 4), (512, 16, 4, 1))
        buf95 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_47], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf91, buf92, primals_95, buf94, primals_97, buf95, 4352, grid=grid(4352), stream=stream0)
        buf138 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_88, input_89], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf94, primals_97, buf138, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 32, 4, 4), (512, 16, 4, 1))
        buf97 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_48], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf91, buf92, primals_95, buf94, primals_97, buf96, primals_99, buf97, 6400, grid=grid(6400), stream=stream0)
        buf137 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_90, input_91], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf96, primals_99, buf137, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_92], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 32, 4, 4), (512, 16, 4, 1))
        buf99 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_49], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf91, buf92, primals_95, buf94, primals_97, buf96, primals_99, buf98, primals_101, buf99, 8448, grid=grid(8448), stream=stream0)
        del buf92
        del buf94
        del buf96
        del primals_95
        del primals_97
        del primals_99
        buf136 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf98, primals_101, buf136, 2048, grid=grid(2048), stream=stream0)
        del buf98
        del primals_101
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 4, 4, 4), (64, 16, 4, 1))
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [input_94, mul_12, out_9], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf101, primals_103, buf91, 256, grid=grid(256), stream=stream0)
        del primals_103
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 32, 4, 4), (512, 16, 4, 1))
        buf103 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_51], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf101, buf102, primals_105, buf103, 2304, grid=grid(2304), stream=stream0)
        buf135 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_95, input_96], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf102, primals_105, buf135, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_97], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 32, 4, 4), (512, 16, 4, 1))
        buf105 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_52], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf101, buf102, primals_105, buf104, primals_107, buf105, 4352, grid=grid(4352), stream=stream0)
        buf134 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_97, input_98], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf104, primals_107, buf134, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 32, 4, 4), (512, 16, 4, 1))
        buf107 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_53], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf101, buf102, primals_105, buf104, primals_107, buf106, primals_109, buf107, 6400, grid=grid(6400), stream=stream0)
        buf133 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_99, input_100], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf106, primals_109, buf133, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_101], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 32, 4, 4), (512, 16, 4, 1))
        buf109 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf101, buf102, primals_105, buf104, primals_107, buf106, primals_109, buf108, primals_111, buf109, 8448, grid=grid(8448), stream=stream0)
        del buf102
        del buf104
        del buf106
        del primals_105
        del primals_107
        del primals_109
        buf132 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_101, input_102], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf108, primals_111, buf132, 2048, grid=grid(2048), stream=stream0)
        del buf108
        del primals_111
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 4, 4, 4), (64, 16, 4, 1))
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [input_103, mul_13, out_10], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf111, primals_113, buf101, 256, grid=grid(256), stream=stream0)
        del primals_113
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 32, 4, 4), (512, 16, 4, 1))
        buf113 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_56], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf111, buf112, primals_115, buf113, 2304, grid=grid(2304), stream=stream0)
        buf131 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_104, input_105], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf112, primals_115, buf131, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_106], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 32, 4, 4), (512, 16, 4, 1))
        buf115 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_57], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf111, buf112, primals_115, buf114, primals_117, buf115, 4352, grid=grid(4352), stream=stream0)
        buf130 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_106, input_107], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf114, primals_117, buf130, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 32, 4, 4), (512, 16, 4, 1))
        buf117 = empty_strided_cuda((4, 100, 4, 4), (1600, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_58], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf111, buf112, primals_115, buf114, primals_117, buf116, primals_119, buf117, 6400, grid=grid(6400), stream=stream0)
        buf129 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_108, input_109], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf116, primals_119, buf129, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_120, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 32, 4, 4), (512, 16, 4, 1))
        buf119 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_59], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf111, buf112, primals_115, buf114, primals_117, buf116, primals_119, buf118, primals_121, buf119, 8448, grid=grid(8448), stream=stream0)
        del buf112
        del buf114
        del buf116
        del primals_115
        del primals_117
        del primals_119
        buf128 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_110, input_111], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_2.run(buf118, primals_121, buf128, 2048, grid=grid(2048), stream=stream0)
        del buf118
        del primals_121
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 4, 4, 4), (64, 16, 4, 1))
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [input_112, mul_14, out_11, mul_15, input_113], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf121, primals_123, buf111, buf91, 256, grid=grid(256), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 4, 4, 4), (64, 16, 4, 1))
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [input_114, x], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_8.run(buf123, primals_125, buf1, 256, grid=grid(256), stream=stream0)
        del primals_125
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_126, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 4, 4, 4), (64, 16, 4, 1))
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [input_115, input_116], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_9.run(buf125, primals_127, 256, grid=grid(256), stream=stream0)
        del primals_127
        # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 4, 4, 4), (64, 16, 4, 1))
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf127, primals_129, 256, grid=grid(256), stream=stream0)
        del primals_129
    return (buf127, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf27, buf29, buf31, buf33, buf35, buf37, buf39, buf41, buf43, buf45, buf47, buf49, buf51, buf53, buf55, buf57, buf59, buf61, buf63, buf65, buf67, buf69, buf71, buf73, buf75, buf77, buf79, buf81, buf83, buf85, buf87, buf89, buf91, buf93, buf95, buf97, buf99, buf101, buf103, buf105, buf107, buf109, buf111, buf113, buf115, buf117, buf119, buf121, buf123, buf125, buf128, buf129, buf130, buf131, buf132, buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, 100, 3, 3), (900, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((4, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
