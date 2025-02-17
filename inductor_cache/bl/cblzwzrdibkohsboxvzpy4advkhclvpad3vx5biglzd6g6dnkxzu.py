# AOT ID: ['19_forward']
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


# kernel path: inductor_cache/oe/coebgleqnmrv5yvsyit4wndarcihe7n3xqgod5gadeskd7qal42x.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=7] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nd/cndk4zcx4rxibuylhrrbo3xoplw7bi6ssclubjsqat6ehytj6tbx.py
# Topologically Sorted Source Nodes: [conv2d_1, x1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
#   x1 => gt, mul, where
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_4, %primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.2), kwargs = {})
#   %where : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_1, %mul), kwargs = {})
#   %gt_41 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 16)
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


# kernel path: inductor_cache/nr/cnrqwv4iguan2tgzdvsi37ti4rf7zcrb5ogayjpotlghdhfathq4.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where], 1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 48)
    x0 = (xindex % 16)
    x2 = xindex // 768
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 512*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 48, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 16*((-32) + x1) + 256*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-32) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: inductor_cache/is/cisyqafvbvbxcqx5ssqqgd6h3x3s2pwexfhipc7e4vrkb6vuopx6.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
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
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 64)
    x0 = (xindex % 16)
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 512*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 48, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-32) + x1) + 256*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-32) + x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.2
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 64, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr3 + (x0 + 16*((-48) + x1) + 256*x2), tmp20, other=0.0)
    tmp24 = tl.load(in_ptr4 + ((-48) + x1), tmp20, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp34, None)
''', device_str='cuda')


# kernel path: inductor_cache/k6/ck6u6cmulmjg2u2a7qyuye5yqe3lzr47l7ndfxf57zqn5qf4xzm6.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
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
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 80)
    x0 = (xindex % 16)
    x2 = xindex // 1280
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 512*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 48, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-32) + x1) + 256*x2), tmp9 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-32) + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.2
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 64, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr3 + (x0 + 16*((-48) + x1) + 256*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.load(in_ptr4 + ((-48) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = 0.0
    tmp28 = tmp26 > tmp27
    tmp29 = 0.2
    tmp30 = tmp26 * tmp29
    tmp31 = tl.where(tmp28, tmp26, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tmp0 >= tmp21
    tmp35 = tl.full([1], 80, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr5 + (x0 + 16*((-64) + x1) + 256*x2), tmp34 & xmask, other=0.0)
    tmp38 = tl.load(in_ptr6 + ((-64) + x1), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: inductor_cache/jd/cjdrbufzivydi62zh4czo22buefxt66to5sthkbw27jpaas4f6b5.py
# Topologically Sorted Source Nodes: [x4, mul, out], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul => mul_3
#   out => add
#   x4 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %primals_10, %primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_4, 0.2), kwargs = {})
#   %add : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %convolution), kwargs = {})
triton_poi_fused_add_convolution_mul_5 = async_compile.triton('triton_poi_fused_add_convolution_mul_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hw/chwp3cqakyo3whisgukdk4epudikj3dko225lvsbjmz7dfpfcrpl.py
# Topologically Sorted Source Nodes: [x4_2, mul_2, out_2, mul_3, input_1], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_1 => add_3
#   mul_2 => mul_11
#   mul_3 => mul_12
#   out_2 => add_2
#   x4_2 => convolution_12
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_8, %primals_26, %primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_12, 0.2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %add_1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.2), kwargs = {})
#   %add_3 : [num_users=7] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %convolution), kwargs = {})
triton_poi_fused_add_convolution_mul_6 = async_compile.triton('triton_poi_fused_add_convolution_mul_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 32)
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


# kernel path: inductor_cache/ku/ckur3x2aeaahbh3n4g5i4kzulu6q2k5migbi7oyntfz2ecxtwr4b.py
# Topologically Sorted Source Nodes: [conv2d_26, input_3], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_26 => convolution_26
#   input_3 => relu
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_7, %primals_54, %primals_55, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_26,), kwargs = {})
triton_poi_fused_convolution_relu_7 = async_compile.triton('triton_poi_fused_convolution_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbriivfqlors6mgtyqhr7cd2lpykegubl4aw3ol72sl44di73u2x.py
# Topologically Sorted Source Nodes: [shortcut, conv2d_27, input_4, input_5, x_1], Original ATen: [aten.convolution, aten.relu, aten.add, aten.leaky_relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   conv2d_27 => convolution_27
#   input_4 => relu_1
#   input_5 => add_8
#   shortcut => convolution_25
#   x_1 => gt_18, mul_26, where_18
# Graph fragment:
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_7, %primals_52, %primals_53, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_27 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_56, %primals_57, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_27,), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_1, %convolution_25), kwargs = {})
#   %gt_18 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_8, 0), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 0.01), kwargs = {})
#   %where_18 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_18, %add_8, %mul_26), kwargs = {})
#   %le_4 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_8 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = 0.0
    tmp10 = tmp8 > tmp9
    tmp11 = 0.01
    tmp12 = tmp8 * tmp11
    tmp13 = tl.where(tmp10, tmp8, tmp12)
    tmp14 = tmp4 <= tmp9
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ql/cqllbv2jej4l4uqop5cjlju5pdiebwrt62iwqhdj2u25fqtsabwb.py
# Topologically Sorted Source Nodes: [conv2d_29, input_6], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_29 => convolution_29
#   input_6 => relu_2
# Graph fragment:
#   %convolution_29 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_18, %primals_60, %primals_61, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_29,), kwargs = {})
triton_poi_fused_convolution_relu_9 = async_compile.triton('triton_poi_fused_convolution_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bm/cbmq737zjg4fuckmcjap6c6a2uld4xgrg5aunhpe4plv2nc43g7z.py
# Topologically Sorted Source Nodes: [shortcut_1, conv2d_30, input_7, input_8, x_2], Original ATen: [aten.convolution, aten.relu, aten.add, aten.leaky_relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   conv2d_30 => convolution_30
#   input_7 => relu_3
#   input_8 => add_9
#   shortcut_1 => convolution_28
#   x_2 => gt_19, mul_27, where_19
# Graph fragment:
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_18, %primals_58, %primals_59, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_30 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_62, %primals_63, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_30,), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_3, %convolution_28), kwargs = {})
#   %gt_19 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 0), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 0.01), kwargs = {})
#   %where_19 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_19, %add_9, %mul_27), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_10 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = 0.0
    tmp10 = tmp8 > tmp9
    tmp11 = 0.01
    tmp12 = tmp8 * tmp11
    tmp13 = tl.where(tmp10, tmp8, tmp12)
    tmp14 = tmp4 <= tmp9
    tl.store(out_ptr0 + (x2), tmp10, xmask)
    tl.store(in_out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mr/cmrqnhuyi2glwc5l6nyk2dpisn2wyqjybcrmlr2hfbfxgbm4rcyy.py
# Topologically Sorted Source Nodes: [conv2d_32, input_9], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_32 => convolution_32
#   input_9 => relu_4
# Graph fragment:
#   %convolution_32 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_19, %primals_66, %primals_67, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_32,), kwargs = {})
triton_poi_fused_convolution_relu_11 = async_compile.triton('triton_poi_fused_convolution_relu_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gk/cgkbixnpjsxciw6lgtx3dve6uasdp4mmvfx5an7esf7lr2nrl4f4.py
# Topologically Sorted Source Nodes: [shortcut_2, conv2d_33, input_10, input_11, x_3], Original ATen: [aten.convolution, aten.relu, aten.add, aten.leaky_relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   conv2d_33 => convolution_33
#   input_10 => relu_5
#   input_11 => add_10
#   shortcut_2 => convolution_31
#   x_3 => gt_20, mul_28, where_20
# Graph fragment:
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_19, %primals_64, %primals_65, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_33 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_68, %primals_69, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_33,), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_5, %convolution_31), kwargs = {})
#   %gt_20 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_10, 0), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.01), kwargs = {})
#   %where_20 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_20, %add_10, %mul_28), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_5, 0), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_12 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = 0.0
    tmp10 = tmp8 > tmp9
    tmp11 = 0.01
    tmp12 = tmp8 * tmp11
    tmp13 = tl.where(tmp10, tmp8, tmp12)
    tmp14 = tmp4 <= tmp9
    tl.store(out_ptr0 + (x2), tmp10, xmask)
    tl.store(in_out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp14, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69 = args
    args.clear()
    assert_size_stride(primals_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (16, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (32, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_13, (16, ), (1, ))
    assert_size_stride(primals_14, (16, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (16, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_17, (16, ), (1, ))
    assert_size_stride(primals_18, (32, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_21, (16, ), (1, ))
    assert_size_stride(primals_22, (16, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_23, (16, ), (1, ))
    assert_size_stride(primals_24, (16, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_25, (16, ), (1, ))
    assert_size_stride(primals_26, (32, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_29, (16, ), (1, ))
    assert_size_stride(primals_30, (16, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_31, (16, ), (1, ))
    assert_size_stride(primals_32, (16, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_33, (16, ), (1, ))
    assert_size_stride(primals_34, (32, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_37, (16, ), (1, ))
    assert_size_stride(primals_38, (16, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_39, (16, ), (1, ))
    assert_size_stride(primals_40, (16, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_41, (16, ), (1, ))
    assert_size_stride(primals_42, (32, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_43, (32, ), (1, ))
    assert_size_stride(primals_44, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_45, (16, ), (1, ))
    assert_size_stride(primals_46, (16, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_47, (16, ), (1, ))
    assert_size_stride(primals_48, (16, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_49, (16, ), (1, ))
    assert_size_stride(primals_50, (32, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_51, (32, ), (1, ))
    assert_size_stride(primals_52, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_69, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 32, 4, 4), (512, 16, 4, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf1, primals_2, 2048, grid=grid(2048), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 16, 4, 4), (256, 16, 4, 1))
        buf88 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_1, x1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf2, primals_5, buf88, 1024, grid=grid(1024), stream=stream0)
        buf3 = empty_strided_cuda((4, 48, 4, 4), (768, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf1, buf2, primals_5, buf3, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 16, 4, 4), (256, 16, 4, 1))
        buf87 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_2, x2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf4, primals_7, buf87, 1024, grid=grid(1024), stream=stream0)
        buf5 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf1, buf2, primals_5, buf4, primals_7, buf5, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 16, 4, 4), (256, 16, 4, 1))
        buf86 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_3, x3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf6, primals_9, buf86, 1024, grid=grid(1024), stream=stream0)
        buf7 = empty_strided_cuda((4, 80, 4, 4), (1280, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf1, buf2, primals_5, buf4, primals_7, buf6, primals_9, buf7, 5120, grid=grid(5120), stream=stream0)
        del buf2
        del buf4
        del buf6
        del primals_5
        del primals_7
        del primals_9
        # Topologically Sorted Source Nodes: [x4], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 32, 4, 4), (512, 16, 4, 1))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x4, mul, out], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_5.run(buf9, primals_11, buf1, 2048, grid=grid(2048), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 16, 4, 4), (256, 16, 4, 1))
        buf85 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_5, x1_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf10, primals_13, buf85, 1024, grid=grid(1024), stream=stream0)
        buf11 = empty_strided_cuda((4, 48, 4, 4), (768, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf9, buf10, primals_13, buf11, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 16, 4, 4), (256, 16, 4, 1))
        buf84 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_6, x2_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf12, primals_15, buf84, 1024, grid=grid(1024), stream=stream0)
        buf13 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf9, buf10, primals_13, buf12, primals_15, buf13, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 16, 4, 4), (256, 16, 4, 1))
        buf83 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_7, x3_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf14, primals_17, buf83, 1024, grid=grid(1024), stream=stream0)
        buf15 = empty_strided_cuda((4, 80, 4, 4), (1280, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf9, buf10, primals_13, buf12, primals_15, buf14, primals_17, buf15, 5120, grid=grid(5120), stream=stream0)
        del buf10
        del buf12
        del buf14
        del primals_13
        del primals_15
        del primals_17
        # Topologically Sorted Source Nodes: [x4_1], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 32, 4, 4), (512, 16, 4, 1))
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x4_1, mul_1, out_1], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_5.run(buf17, primals_19, buf9, 2048, grid=grid(2048), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 16, 4, 4), (256, 16, 4, 1))
        buf82 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_9, x1_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf18, primals_21, buf82, 1024, grid=grid(1024), stream=stream0)
        buf19 = empty_strided_cuda((4, 48, 4, 4), (768, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf17, buf18, primals_21, buf19, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 16, 4, 4), (256, 16, 4, 1))
        buf81 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_10, x2_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf20, primals_23, buf81, 1024, grid=grid(1024), stream=stream0)
        buf21 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf17, buf18, primals_21, buf20, primals_23, buf21, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 16, 4, 4), (256, 16, 4, 1))
        buf80 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_11, x3_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf22, primals_25, buf80, 1024, grid=grid(1024), stream=stream0)
        buf23 = empty_strided_cuda((4, 80, 4, 4), (1280, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf17, buf18, primals_21, buf20, primals_23, buf22, primals_25, buf23, 5120, grid=grid(5120), stream=stream0)
        del buf18
        del buf20
        del buf22
        del primals_21
        del primals_23
        del primals_25
        # Topologically Sorted Source Nodes: [x4_2], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 32, 4, 4), (512, 16, 4, 1))
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x4_2, mul_2, out_2, mul_3, input_1], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf25, primals_27, buf17, buf1, 2048, grid=grid(2048), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 16, 4, 4), (256, 16, 4, 1))
        buf79 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_13, x1_3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf26, primals_29, buf79, 1024, grid=grid(1024), stream=stream0)
        buf27 = empty_strided_cuda((4, 48, 4, 4), (768, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf25, buf26, primals_29, buf27, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 16, 4, 4), (256, 16, 4, 1))
        buf78 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_14, x2_3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf28, primals_31, buf78, 1024, grid=grid(1024), stream=stream0)
        buf29 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf25, buf26, primals_29, buf28, primals_31, buf29, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 16, 4, 4), (256, 16, 4, 1))
        buf77 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_15, x3_3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf30, primals_33, buf77, 1024, grid=grid(1024), stream=stream0)
        buf31 = empty_strided_cuda((4, 80, 4, 4), (1280, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf25, buf26, primals_29, buf28, primals_31, buf30, primals_33, buf31, 5120, grid=grid(5120), stream=stream0)
        del buf26
        del buf28
        del buf30
        del primals_29
        del primals_31
        del primals_33
        # Topologically Sorted Source Nodes: [x4_3], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 32, 4, 4), (512, 16, 4, 1))
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x4_3, mul_4, out_3], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_5.run(buf33, primals_35, buf25, 2048, grid=grid(2048), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 16, 4, 4), (256, 16, 4, 1))
        buf76 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_17, x1_4], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf34, primals_37, buf76, 1024, grid=grid(1024), stream=stream0)
        buf35 = empty_strided_cuda((4, 48, 4, 4), (768, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf33, buf34, primals_37, buf35, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 16, 4, 4), (256, 16, 4, 1))
        buf75 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_18, x2_4], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf36, primals_39, buf75, 1024, grid=grid(1024), stream=stream0)
        buf37 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf33, buf34, primals_37, buf36, primals_39, buf37, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 16, 4, 4), (256, 16, 4, 1))
        buf74 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_19, x3_4], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf38, primals_41, buf74, 1024, grid=grid(1024), stream=stream0)
        buf39 = empty_strided_cuda((4, 80, 4, 4), (1280, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf33, buf34, primals_37, buf36, primals_39, buf38, primals_41, buf39, 5120, grid=grid(5120), stream=stream0)
        del buf34
        del buf36
        del buf38
        del primals_37
        del primals_39
        del primals_41
        # Topologically Sorted Source Nodes: [x4_4], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 32, 4, 4), (512, 16, 4, 1))
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x4_4, mul_5, out_4], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_5.run(buf41, primals_43, buf33, 2048, grid=grid(2048), stream=stream0)
        del primals_43
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 16, 4, 4), (256, 16, 4, 1))
        buf73 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_21, x1_5], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf42, primals_45, buf73, 1024, grid=grid(1024), stream=stream0)
        buf43 = empty_strided_cuda((4, 48, 4, 4), (768, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf41, buf42, primals_45, buf43, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 16, 4, 4), (256, 16, 4, 1))
        buf72 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_22, x2_5], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf44, primals_47, buf72, 1024, grid=grid(1024), stream=stream0)
        buf45 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf41, buf42, primals_45, buf44, primals_47, buf45, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 16, 4, 4), (256, 16, 4, 1))
        buf71 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_23, x3_5], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1.run(buf46, primals_49, buf71, 1024, grid=grid(1024), stream=stream0)
        buf47 = empty_strided_cuda((4, 80, 4, 4), (1280, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf41, buf42, primals_45, buf44, primals_47, buf46, primals_49, buf47, 5120, grid=grid(5120), stream=stream0)
        del buf42
        del buf44
        del buf46
        del primals_45
        del primals_47
        del primals_49
        # Topologically Sorted Source Nodes: [x4_5], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 32, 4, 4), (512, 16, 4, 1))
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x4_5, mul_6, out_5, mul_7, input_2], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf49, primals_51, buf41, buf25, 2048, grid=grid(2048), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [shortcut], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_52, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 64, 2, 2), (256, 4, 2, 1))
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf49, primals_54, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 64, 2, 2), (256, 4, 2, 1))
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [conv2d_26, input_3], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_7.run(buf52, primals_55, 1024, grid=grid(1024), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 64, 2, 2), (256, 4, 2, 1))
        buf54 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.bool)
        buf55 = buf50; del buf50  # reuse
        buf70 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.bool)
        # Topologically Sorted Source Nodes: [shortcut, conv2d_27, input_4, input_5, x_1], Original ATen: [aten.convolution, aten.relu, aten.add, aten.leaky_relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_8.run(buf55, buf53, primals_57, primals_53, buf54, buf70, 1024, grid=grid(1024), stream=stream0)
        del buf53
        del primals_53
        del primals_57
        # Topologically Sorted Source Nodes: [shortcut_1], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_58, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 128, 1, 1), (128, 1, 1, 1))
        # Topologically Sorted Source Nodes: [conv2d_29], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf55, primals_60, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 128, 1, 1), (128, 1, 1, 1))
        buf58 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [conv2d_29, input_6], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_9.run(buf58, primals_61, 512, grid=grid(512), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [conv2d_30], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 128, 1, 1), (128, 1, 1, 1))
        buf60 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.bool)
        buf61 = buf56; del buf56  # reuse
        buf69 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [shortcut_1, conv2d_30, input_7, input_8, x_2], Original ATen: [aten.convolution, aten.relu, aten.add, aten.leaky_relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_10.run(buf61, buf59, primals_63, primals_59, buf60, buf69, 512, grid=grid(512), stream=stream0)
        del buf59
        del primals_59
        del primals_63
        # Topologically Sorted Source Nodes: [shortcut_2], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_64, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 256, 1, 1), (256, 1, 1, 1))
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf61, primals_66, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 256, 1, 1), (256, 1, 1, 1))
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [conv2d_32, input_9], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_11.run(buf64, primals_67, 1024, grid=grid(1024), stream=stream0)
        del primals_67
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 256, 1, 1), (256, 1, 1, 1))
        buf66 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.bool)
        buf67 = buf62; del buf62  # reuse
        buf68 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [shortcut_2, conv2d_33, input_10, input_11, x_3], Original ATen: [aten.convolution, aten.relu, aten.add, aten.leaky_relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_relu_threshold_backward_12.run(buf67, buf65, primals_69, primals_65, buf66, buf68, 1024, grid=grid(1024), stream=stream0)
        del buf65
        del primals_65
        del primals_69
    return (buf55, buf61, buf67, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf27, buf29, buf31, buf33, buf35, buf37, buf39, buf41, buf43, buf45, buf47, buf49, buf52, buf54, buf55, buf58, buf60, buf61, buf64, buf66, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((16, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((16, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((16, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((16, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((16, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((16, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((32, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((16, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((16, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
