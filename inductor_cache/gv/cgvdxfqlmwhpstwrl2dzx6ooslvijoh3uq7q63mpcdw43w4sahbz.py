# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/qi/cqiqse22dupydkpgnvmscg2zppmnoqbrzvui3pkjje4qqhf7kkok.py
# Topologically Sorted Source Nodes: [conv2d_1, add, x], Original ATen: [aten.convolution, aten.add, aten.leaky_relu]
# Source node to ATen node mapping:
#   add => add
#   conv2d_1 => convolution_1
#   x => gt, mul, where
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_4, %primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %primals_3), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.05), kwargs = {})
#   %where : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add, %mul), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_0 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 50)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = 0.05
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crzkyvluxge7v5kltrmx56d2bszwyofx6yy7ewfor6xkj77o7qhl.py
# Topologically Sorted Source Nodes: [cat, x_4], Original ATen: [aten.cat, aten.leaky_relu]
# Source node to ATen node mapping:
#   cat => cat
#   x_4 => gt_3, mul_3, where_3
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6], 1), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%cat, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, 0.05), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %cat, %mul_3), kwargs = {})
triton_poi_fused_cat_leaky_relu_1 = async_compile.triton('triton_poi_fused_cat_leaky_relu_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_leaky_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_leaky_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 100)
    x0 = (xindex % 4096)
    x2 = xindex // 409600
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 25, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 102400*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 50, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 4096*((-25) + x1) + 102400*x2), tmp13, other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-25) + x1), tmp13, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 75, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr4 + (x0 + 4096*((-50) + x1) + 102400*x2), tmp22, other=0.0)
    tmp24 = tl.load(in_ptr5 + ((-50) + x1), tmp22, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tmp0 >= tmp20
    tmp29 = tl.full([1], 100, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr6 + (x0 + 4096*((-75) + x1) + 102400*x2), tmp28, other=0.0)
    tmp32 = tl.load(in_ptr7 + ((-75) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp28, tmp33, tmp34)
    tmp36 = tl.where(tmp22, tmp27, tmp35)
    tmp37 = tl.where(tmp13, tmp18, tmp36)
    tmp38 = tl.where(tmp4, tmp9, tmp37)
    tmp39 = 0.0
    tmp40 = tmp38 > tmp39
    tmp41 = 0.05
    tmp42 = tmp38 * tmp41
    tmp43 = tl.where(tmp40, tmp38, tmp42)
    tl.store(in_out_ptr0 + (x3), tmp43, None)
''', device_str='cuda')


# kernel path: inductor_cache/ni/cniymh5svafvk53xvr6mh6sczmjnenkhrg7u4s3wfbrdx5otd5wk.py
# Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_7 => convolution_7
# Graph fragment:
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_3, %primals_16, %primals_17, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 50)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/xp/cxp4dd6jjhveqrichsracmisk5z32eueg3sdrsqupsurdstvh6go.py
# Topologically Sorted Source Nodes: [x1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x1 => convolution_8
# Graph fragment:
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_7, %primals_18, %primals_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 12)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/sz/cszxxmns7e4rj5ummoxxpmjsd65somrb54nmt5owchmbru7qpl4v.py
# Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_9 => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_8, %primals_20, %primals_21, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_4 = async_compile.triton('triton_poi_fused_convolution_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 46128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 961) % 12)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hc/chcoh642kkq7s5bftgkee3tmu3nooy26uvo5jzpyjc4vypf3clzo.py
# Topologically Sorted Source Nodes: [conv2d_10, x2_1], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_10 => convolution_10
#   x2_1 => relu
# Graph fragment:
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_22, %primals_23, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
triton_poi_fused_convolution_relu_5 = async_compile.triton('triton_poi_fused_convolution_relu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 81) % 12)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wf/cwfuighajtkixuifcpt5zri4wxq6u3voasrdmpga3epr3psq5ews.py
# Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x2_3 => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.140625
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xn/cxnzwepxe3up4det3kyy6unhcgyspq4ubzibby2gc36vxo6bjuzl.py
# Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x2_3 => add_4, clamp_max
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_4, 8), kwargs = {})
triton_poi_fused_add_clamp_7 = async_compile.triton('triton_poi_fused_add_clamp_7', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.140625
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 8, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ww/cwwhurkozbsrqrrnom47xonjytbvbacvcisgwcu6xuiazsyohneh.py
# Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   x2_3 => add_3, clamp_max_2, clamp_min, clamp_min_2, convert_element_type, iota, mul_4, sub, sub_2
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.140625), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_4, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.140625
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gd/cgdxed35lvbouqx2es56d6m2icbxrapg7vgx7tco4j4lppkuo7gs.py
# Topologically Sorted Source Nodes: [conv2d_12, x2_3, conv2d_13, add_3], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_10
#   conv2d_12 => convolution_12
#   conv2d_13 => convolution_13
#   x2_3 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_7, add_8, add_9, mul_6, mul_7, mul_8, sub_3, sub_4, sub_6
# Graph fragment:
#   %convolution_12 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_26, %primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_12, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_12, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_12, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_12, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_6), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_7), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %add_7), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %mul_8), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_8, %primals_28, %primals_29, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convolution_13), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_sub_9 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_sub_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_sub_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_sub_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex // 4096
    x2 = ((xindex // 4096) % 12)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x6), None)
    tmp38 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 9, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 9*tmp4 + 81*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 + tmp1
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tmp16 = tl.load(in_ptr2 + (tmp15 + 9*tmp4 + 81*x5), None, eviction_policy='evict_last')
    tmp17 = tmp16 + tmp10
    tmp18 = tmp17 - tmp11
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 + tmp20
    tmp23 = tmp22 + tmp1
    tmp24 = tmp22 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tmp26 = tl.load(in_ptr2 + (tmp8 + 9*tmp25 + 81*x5), None, eviction_policy='evict_last')
    tmp27 = tmp26 + tmp10
    tmp28 = tl.load(in_ptr2 + (tmp15 + 9*tmp25 + 81*x5), None, eviction_policy='evict_last')
    tmp29 = tmp28 + tmp10
    tmp30 = tmp29 - tmp27
    tmp31 = tmp30 * tmp19
    tmp32 = tmp27 + tmp31
    tmp33 = tmp32 - tmp21
    tmp35 = tmp33 * tmp34
    tmp36 = tmp21 + tmp35
    tmp39 = tmp37 + tmp38
    tmp40 = tmp36 + tmp39
    tl.store(in_out_ptr0 + (x6), tmp40, None)
''', device_str='cuda')


# kernel path: inductor_cache/we/cwesgg2hwlqhwmms32mm572te3gxzaycfafgsozacrnshsom2oxt.py
# Topologically Sorted Source Nodes: [x2_4, sigmoid, x_5], Original ATen: [aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
#   x2_4 => convolution_14
#   x_5 => mul_9
# Graph fragment:
#   %convolution_14 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_10, %primals_30, %primals_31, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_14,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_7, %sigmoid), kwargs = {})
triton_poi_fused_convolution_mul_sigmoid_10 = async_compile.triton('triton_poi_fused_convolution_mul_sigmoid_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sigmoid_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_10(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 50)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sigmoid(tmp2)
    tmp5 = tmp3 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31 = args
    args.clear()
    assert_size_stride(primals_1, (25, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(primals_2, (25, ), (1, ))
    assert_size_stride(primals_3, (4, 50, 64, 64), (204800, 4096, 64, 1))
    assert_size_stride(primals_4, (50, 50, 3, 3), (450, 9, 3, 1))
    assert_size_stride(primals_5, (50, ), (1, ))
    assert_size_stride(primals_6, (25, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(primals_7, (25, ), (1, ))
    assert_size_stride(primals_8, (50, 50, 3, 3), (450, 9, 3, 1))
    assert_size_stride(primals_9, (50, ), (1, ))
    assert_size_stride(primals_10, (25, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(primals_11, (25, ), (1, ))
    assert_size_stride(primals_12, (50, 50, 3, 3), (450, 9, 3, 1))
    assert_size_stride(primals_13, (50, ), (1, ))
    assert_size_stride(primals_14, (25, 50, 3, 3), (450, 9, 3, 1))
    assert_size_stride(primals_15, (25, ), (1, ))
    assert_size_stride(primals_16, (50, 100, 1, 1), (100, 1, 1, 1))
    assert_size_stride(primals_17, (50, ), (1, ))
    assert_size_stride(primals_18, (12, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(primals_19, (12, ), (1, ))
    assert_size_stride(primals_20, (12, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_21, (12, ), (1, ))
    assert_size_stride(primals_22, (12, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_23, (12, ), (1, ))
    assert_size_stride(primals_24, (12, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_25, (12, ), (1, ))
    assert_size_stride(primals_26, (12, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_27, (12, ), (1, ))
    assert_size_stride(primals_28, (12, 12, 1, 1), (12, 1, 1, 1))
    assert_size_stride(primals_29, (12, ), (1, ))
    assert_size_stride(primals_30, (50, 12, 1, 1), (12, 1, 1, 1))
    assert_size_stride(primals_31, (50, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [d1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 25, 64, 64), (102400, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_3, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 50, 64, 64), (204800, 4096, 64, 1))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1, add, x], Original ATen: [aten.convolution, aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_0.run(buf2, primals_5, primals_3, 819200, grid=grid(819200), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [d2], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 25, 64, 64), (102400, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 50, 64, 64), (204800, 4096, 64, 1))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [conv2d_3, add_1, x_1], Original ATen: [aten.convolution, aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_0.run(buf5, primals_9, buf2, 819200, grid=grid(819200), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [d3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 25, 64, 64), (102400, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf5, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 50, 64, 64), (204800, 4096, 64, 1))
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [conv2d_5, add_2, x_2], Original ATen: [aten.convolution, aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_0.run(buf8, primals_13, buf5, 819200, grid=grid(819200), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 25, 64, 64), (102400, 4096, 64, 1))
        buf10 = empty_strided_cuda((4, 100, 64, 64), (409600, 4096, 64, 1), torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [cat, x_4], Original ATen: [aten.cat, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_leaky_relu_1.run(buf11, buf0, primals_2, buf3, primals_7, buf6, primals_11, buf9, primals_15, 1638400, grid=grid(1638400), stream=stream0)
        del buf0
        del buf3
        del buf6
        del buf9
        del primals_11
        del primals_15
        del primals_2
        del primals_7
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 50, 64, 64), (204800, 4096, 64, 1))
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf13, primals_17, 819200, grid=grid(819200), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [x1], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf15, primals_19, 196608, grid=grid(196608), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_20, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 12, 31, 31), (11532, 961, 31, 1))
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_4.run(buf17, primals_21, 46128, grid=grid(46128), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [x2], Original ATen: [aten.max_pool2d_with_indices]
        buf18 = torch.ops.aten.max_pool2d_with_indices.default(buf17, [7, 7], [3, 3])
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf19, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 12, 9, 9), (972, 81, 9, 1))
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [conv2d_10, x2_1], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_5.run(buf22, primals_23, 3888, grid=grid(3888), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 12, 9, 9), (972, 81, 9, 1))
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [conv2d_11, x2_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_5.run(buf24, primals_25, 3888, grid=grid(3888), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 12, 9, 9), (972, 81, 9, 1))
        buf26 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf26, 64, grid=grid(64), stream=stream0)
        buf27 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_7.run(buf27, 64, grid=grid(64), stream=stream0)
        buf28 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf28, 64, grid=grid(64), stream=stream0)
        buf29 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_7.run(buf29, 64, grid=grid(64), stream=stream0)
        buf30 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8.run(buf30, 64, grid=grid(64), stream=stream0)
        buf32 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8.run(buf32, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf15, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf31 = empty_strided_cuda((4, 12, 64, 64), (49152, 4096, 64, 1), torch.float32)
        buf35 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [conv2d_12, x2_3, conv2d_13, add_3], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_sub_9.run(buf35, buf26, buf28, buf25, primals_27, buf29, buf30, buf27, buf32, buf34, primals_29, 196608, grid=grid(196608), stream=stream0)
        del buf25
        del buf34
        del primals_27
        del primals_29
        # Topologically Sorted Source Nodes: [x2_4], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 50, 64, 64), (204800, 4096, 64, 1))
        buf37 = buf36; del buf36  # reuse
        buf38 = empty_strided_cuda((4, 50, 64, 64), (204800, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x2_4, sigmoid, x_5], Original ATen: [aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_10.run(buf37, primals_31, buf13, buf38, 819200, grid=grid(819200), stream=stream0)
        del primals_31
    return (buf38, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, buf2, buf5, buf8, buf11, buf13, buf15, buf17, buf19, buf20, buf22, buf24, buf26, buf27, buf28, buf29, buf30, buf32, buf35, buf37, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((25, 50, 1, 1), (50, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 50, 64, 64), (204800, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((50, 50, 3, 3), (450, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((25, 50, 1, 1), (50, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((50, 50, 3, 3), (450, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((25, 50, 1, 1), (50, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((50, 50, 3, 3), (450, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((25, 50, 3, 3), (450, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((50, 100, 1, 1), (100, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((12, 50, 1, 1), (50, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((12, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((12, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((12, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((12, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((12, 12, 1, 1), (12, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((50, 12, 1, 1), (12, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
