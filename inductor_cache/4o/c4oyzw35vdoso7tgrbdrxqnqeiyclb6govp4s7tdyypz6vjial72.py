# AOT ID: ['4_inference']
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


# kernel path: inductor_cache/xu/cxupsvzkg5j4u462qod2vqq7fcehn5zmjklleqfcpyrhx2w7gld6.py
# Topologically Sorted Source Nodes: [augmented, aug_prediction, flip_1, sum_1], Original ATen: [aten.cat, aten.relu, aten.flip, aten.sum]
# Source node to ATen node mapping:
#   aug_prediction => relu
#   augmented => cat
#   flip_1 => rev_1
#   sum_1 => sum_1
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1],), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%cat,), kwargs = {})
#   %rev_1 : [num_users=1] = call_function[target=torch.ops.prims.rev.default](args = (%select_4, [2]), kwargs = {})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%relu, %rev_1, 0, 1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%select_scatter_default, [0]), kwargs = {})
triton_poi_fused_cat_flip_relu_sum_0 = async_compile.triton('triton_poi_fused_cat_flip_relu_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_flip_relu_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_flip_relu_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp3 < tmp3
    tmp7 = tl.load(in_ptr0 + (3 + ((-1)*x0) + 4*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp3 >= tmp3
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp3 < tmp9
    tmp11 = tl.load(in_ptr0 + (x2), tmp8 & xmask, other=0.0)
    tmp12 = tl.where(tmp6, tmp7, tmp11)
    tmp13 = triton_helpers.maximum(tmp0, tmp12)
    tmp14 = tmp4 >= tmp4
    tmp15 = tmp4 < tmp3
    tmp16 = tl.load(in_ptr0 + (x2), tmp15 & xmask, other=0.0)
    tmp17 = tmp4 >= tmp3
    tmp18 = tmp4 < tmp9
    tmp19 = tl.load(in_ptr0 + (3 + ((-1)*x0) + 4*x1), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp15, tmp16, tmp19)
    tmp21 = triton_helpers.maximum(tmp0, tmp20)
    tmp22 = tl.where(tmp2, tmp13, tmp21)
    tmp23 = tmp1 == tmp1
    tmp24 = tl.load(in_ptr0 + (x2), tmp6 & xmask, other=0.0)
    tmp25 = tl.load(in_ptr0 + (3 + ((-1)*x0) + 4*x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = triton_helpers.maximum(tmp0, tmp26)
    tmp28 = tl.where(tmp23, tmp13, tmp27)
    tmp29 = tmp22 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ju/cjuqqtbai7hgo5da4likezbci2znoly6jxqebprmowbbiezke547.py
# Topologically Sorted Source Nodes: [augmented_1, aug_prediction_1, flip_3, sum_2], Original ATen: [aten.cat, aten.relu, aten.flip, aten.sum]
# Source node to ATen node mapping:
#   aug_prediction_1 => relu_1
#   augmented_1 => cat_1
#   flip_3 => rev_3
#   sum_2 => sum_2
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_3, %unsqueeze_4],), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%cat_1,), kwargs = {})
#   %rev_3 : [num_users=1] = call_function[target=torch.ops.prims.rev.default](args = (%select_7, [2]), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%relu_1, %rev_3, 0, 1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%select_scatter_default_1, [0]), kwargs = {})
triton_poi_fused_cat_flip_relu_sum_1 = async_compile.triton('triton_poi_fused_cat_flip_relu_sum_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_flip_relu_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_flip_relu_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp3 < tmp3
    tmp7 = tl.load(in_ptr0 + (67 + ((-1)*x0) + 4*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp3 >= tmp3
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp3 < tmp9
    tmp11 = tl.load(in_ptr0 + (64 + x2), tmp8 & xmask, other=0.0)
    tmp12 = tl.where(tmp6, tmp7, tmp11)
    tmp13 = triton_helpers.maximum(tmp0, tmp12)
    tmp14 = tmp4 >= tmp4
    tmp15 = tmp4 < tmp3
    tmp16 = tl.load(in_ptr0 + (64 + x2), tmp15 & xmask, other=0.0)
    tmp17 = tmp4 >= tmp3
    tmp18 = tmp4 < tmp9
    tmp19 = tl.load(in_ptr0 + (67 + ((-1)*x0) + 4*x1), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp15, tmp16, tmp19)
    tmp21 = triton_helpers.maximum(tmp0, tmp20)
    tmp22 = tl.where(tmp2, tmp13, tmp21)
    tmp23 = tmp1 == tmp1
    tmp24 = tl.load(in_ptr0 + (64 + x2), tmp6 & xmask, other=0.0)
    tmp25 = tl.load(in_ptr0 + (67 + ((-1)*x0) + 4*x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = triton_helpers.maximum(tmp0, tmp26)
    tmp28 = tl.where(tmp23, tmp13, tmp27)
    tmp29 = tmp22 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wv/cwvgf4g7y5l5dxqwncrczp744eaccxjafanber73g5fqvf23vg4i.py
# Topologically Sorted Source Nodes: [augmented_2, aug_prediction_2, flip_5, sum_3], Original ATen: [aten.cat, aten.relu, aten.flip, aten.sum]
# Source node to ATen node mapping:
#   aug_prediction_2 => relu_2
#   augmented_2 => cat_2
#   flip_5 => rev_5
#   sum_3 => sum_3
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_6, %unsqueeze_7],), kwargs = {})
#   %relu_2 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%cat_2,), kwargs = {})
#   %rev_5 : [num_users=1] = call_function[target=torch.ops.prims.rev.default](args = (%select_10, [2]), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%relu_2, %rev_5, 0, 1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%select_scatter_default_2, [0]), kwargs = {})
triton_poi_fused_cat_flip_relu_sum_2 = async_compile.triton('triton_poi_fused_cat_flip_relu_sum_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_flip_relu_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_flip_relu_sum_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp3 < tmp3
    tmp7 = tl.load(in_ptr0 + (131 + ((-1)*x0) + 4*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp3 >= tmp3
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp3 < tmp9
    tmp11 = tl.load(in_ptr0 + (128 + x2), tmp8 & xmask, other=0.0)
    tmp12 = tl.where(tmp6, tmp7, tmp11)
    tmp13 = triton_helpers.maximum(tmp0, tmp12)
    tmp14 = tmp4 >= tmp4
    tmp15 = tmp4 < tmp3
    tmp16 = tl.load(in_ptr0 + (128 + x2), tmp15 & xmask, other=0.0)
    tmp17 = tmp4 >= tmp3
    tmp18 = tmp4 < tmp9
    tmp19 = tl.load(in_ptr0 + (131 + ((-1)*x0) + 4*x1), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp15, tmp16, tmp19)
    tmp21 = triton_helpers.maximum(tmp0, tmp20)
    tmp22 = tl.where(tmp2, tmp13, tmp21)
    tmp23 = tmp1 == tmp1
    tmp24 = tl.load(in_ptr0 + (128 + x2), tmp6 & xmask, other=0.0)
    tmp25 = tl.load(in_ptr0 + (131 + ((-1)*x0) + 4*x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = triton_helpers.maximum(tmp0, tmp26)
    tmp28 = tl.where(tmp23, tmp13, tmp27)
    tmp29 = tmp22 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vo/cvo4u5snpzkjmuzibzmdsgmnl2b7nrp4fobkvedc4lvnhgpcd7lx.py
# Topologically Sorted Source Nodes: [augmented_3, aug_prediction_3, flip_7, sum_4], Original ATen: [aten.cat, aten.relu, aten.flip, aten.sum]
# Source node to ATen node mapping:
#   aug_prediction_3 => relu_3
#   augmented_3 => cat_3
#   flip_7 => rev_7
#   sum_4 => sum_4
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_9, %unsqueeze_10],), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%cat_3,), kwargs = {})
#   %rev_7 : [num_users=1] = call_function[target=torch.ops.prims.rev.default](args = (%select_13, [2]), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%relu_3, %rev_7, 0, 1), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%select_scatter_default_3, [0]), kwargs = {})
triton_poi_fused_cat_flip_relu_sum_3 = async_compile.triton('triton_poi_fused_cat_flip_relu_sum_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_flip_relu_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_flip_relu_sum_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp3 < tmp3
    tmp7 = tl.load(in_ptr0 + (195 + ((-1)*x0) + 4*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp3 >= tmp3
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp3 < tmp9
    tmp11 = tl.load(in_ptr0 + (192 + x2), tmp8 & xmask, other=0.0)
    tmp12 = tl.where(tmp6, tmp7, tmp11)
    tmp13 = triton_helpers.maximum(tmp0, tmp12)
    tmp14 = tmp4 >= tmp4
    tmp15 = tmp4 < tmp3
    tmp16 = tl.load(in_ptr0 + (192 + x2), tmp15 & xmask, other=0.0)
    tmp17 = tmp4 >= tmp3
    tmp18 = tmp4 < tmp9
    tmp19 = tl.load(in_ptr0 + (195 + ((-1)*x0) + 4*x1), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp15, tmp16, tmp19)
    tmp21 = triton_helpers.maximum(tmp0, tmp20)
    tmp22 = tl.where(tmp2, tmp13, tmp21)
    tmp23 = tmp1 == tmp1
    tmp24 = tl.load(in_ptr0 + (192 + x2), tmp6 & xmask, other=0.0)
    tmp25 = tl.load(in_ptr0 + (195 + ((-1)*x0) + 4*x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = triton_helpers.maximum(tmp0, tmp26)
    tmp28 = tl.where(tmp23, tmp13, tmp27)
    tmp29 = tmp22 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nv/cnva7xjaeddzru6kg2o67lkoogld7zfmqdls4ehqlm2z72w5bc7u.py
# Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_4 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_2, %unsqueeze_5, %unsqueeze_8, %unsqueeze_11],), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.25
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 2, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + (x0), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 0.25
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 3, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr2 + (x0), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = 0.25
    tmp25 = tmp23 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tmp0 >= tmp20
    tmp29 = tl.full([1], 4, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr3 + (x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 0.25
    tmp33 = tmp31 * tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp28, tmp33, tmp34)
    tmp36 = tl.where(tmp22, tmp27, tmp35)
    tmp37 = tl.where(tmp13, tmp18, tmp36)
    tmp38 = tl.where(tmp4, tmp9, tmp37)
    tl.store(out_ptr0 + (x2), tmp38, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [augmented, aug_prediction, flip_1, sum_1], Original ATen: [aten.cat, aten.relu, aten.flip, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_flip_relu_sum_0.run(arg0_1, buf0, 64, grid=grid(64), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [augmented_1, aug_prediction_1, flip_3, sum_2], Original ATen: [aten.cat, aten.relu, aten.flip, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_flip_relu_sum_1.run(arg0_1, buf1, 64, grid=grid(64), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [augmented_2, aug_prediction_2, flip_5, sum_3], Original ATen: [aten.cat, aten.relu, aten.flip, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_flip_relu_sum_2.run(arg0_1, buf2, 64, grid=grid(64), stream=stream0)
        buf3 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [augmented_3, aug_prediction_3, flip_7, sum_4], Original ATen: [aten.cat, aten.relu, aten.flip, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_flip_relu_sum_3.run(arg0_1, buf3, 64, grid=grid(64), stream=stream0)
        del arg0_1
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf0, buf1, buf2, buf3, buf4, 256, grid=grid(256), stream=stream0)
        del buf0
        del buf1
        del buf2
        del buf3
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
