# AOT ID: ['10_inference']
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


# kernel path: inductor_cache/44/c44b57zbzv46krbnpqjub2uyqgqhk5hunawqeucaoqslkls4xpmv.py
# Topologically Sorted Source Nodes: [new_locs_2], Original ATen: [aten.index]
# Source node to ATen node mapping:
#   new_locs_2 => index
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_1, [None, None, None, %lift_fresh_copy]), kwargs = {})
triton_poi_fused_index_0 = async_compile.triton('triton_poi_fused_index_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 16)
    x2 = xindex // 32
    x3 = xindex
    tmp11 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x1 + 32*x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (16 + x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (16 + x1 + 32*x2), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp7 == tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = 0.3333333333333333
    tmp15 = tmp13 * tmp14
    tmp16 = 0.5
    tmp17 = tmp15 - tmp16
    tmp18 = 2.0
    tmp19 = tmp17 * tmp18
    tmp22 = tmp20 + tmp21
    tmp23 = tl.where(tmp10, tmp19, tmp22)
    tmp24 = tmp23 * tmp14
    tmp25 = tmp24 - tmp16
    tmp26 = tmp25 * tmp18
    tmp27 = tmp6 == tmp9
    tmp28 = tl.load(in_ptr0 + (x1 + 16*tmp4), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (x1 + 16*tmp4 + 32*x2), xmask, eviction_policy='evict_last')
    tmp30 = tmp28 + tmp29
    tmp31 = tl.where(tmp27, tmp19, tmp30)
    tmp32 = tl.where(tmp8, tmp26, tmp31)
    tl.store(out_ptr0 + (x3), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r7/cr73h5yd2infalhxg5wygdomfaoa6o4exozrhass7nuhcf53huik.py
# Topologically Sorted Source Nodes: [grid_sample], Original ATen: [aten.grid_sampler_2d]
# Source node to ATen node mapping:
#   grid_sample => add_1, add_2, add_3, add_4, add_5, add_6, add_7, clamp_max, clamp_max_1, clamp_min, clamp_min_1, floor, floor_1, full_default_11, full_default_2, full_default_5, full_default_8, ge, ge_1, ge_2, ge_3, ge_4, ge_5, ge_6, ge_7, index_1, index_2, index_3, index_4, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_3, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt, lt_1, lt_2, lt_3, lt_4, lt_5, lt_6, lt_7, mul_10, mul_11, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9, where_11, where_2, where_5, where_8
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, 2.0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 1.5), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0), kwargs = {})
#   %clamp_max : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 3), kwargs = {})
#   %floor : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%clamp_max,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, 2.0), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 1.5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_2, 0), kwargs = {})
#   %clamp_max_1 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 3), kwargs = {})
#   %floor_1 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%clamp_max_1,), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_1, %lt_1), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt, %logical_and), kwargs = {})
#   %logical_and_2 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %logical_and_1), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%view_1, %view_2, %where_1, %where]), kwargs = {})
#   %add_3 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor, 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %clamp_max), kwargs = {})
#   %add_4 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_1, 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %clamp_max_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %sub_3), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %mul_4, %full_default_2), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_1, %where_2), kwargs = {})
#   %ge_2 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_3, 0), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_3, 4), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_3 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_3, %lt_3), kwargs = {})
#   %logical_and_4 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_2, %logical_and_3), kwargs = {})
#   %logical_and_5 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_2, %logical_and_4), kwargs = {})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%view_1, %view_2, %where_4, %where_3]), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, %floor), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %clamp_max_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %sub_5), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_5, %mul_5, %full_default_5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_2, %where_5), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_4, 0), kwargs = {})
#   %lt_5 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_4, 4), kwargs = {})
#   %logical_and_6 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_5, %lt_5), kwargs = {})
#   %logical_and_7 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_4, %logical_and_6), kwargs = {})
#   %logical_and_8 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_4, %logical_and_7), kwargs = {})
#   %index_3 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%view_1, %view_2, %where_7, %where_6]), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %clamp_max), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max_1, %floor_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %sub_7), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_8, %mul_6, %full_default_8), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_3, %where_8), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_10), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_3, 0), kwargs = {})
#   %lt_6 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_3, 4), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_4, 0), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_4, 4), kwargs = {})
#   %logical_and_9 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_7, %lt_7), kwargs = {})
#   %logical_and_10 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_6, %logical_and_9), kwargs = {})
#   %logical_and_11 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_6, %logical_and_10), kwargs = {})
#   %index_4 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%view_1, %view_2, %where_10, %where_9]), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, %floor), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max_1, %floor_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %sub_9), kwargs = {})
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_11, %mul_7, %full_default_11), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_4, %where_11), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mul_11), kwargs = {})
triton_poi_fused_grid_sampler_2d_1 = async_compile.triton('triton_poi_fused_grid_sampler_2d_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    x4 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1.5
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 3.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = libdevice.floor(tmp8)
    tmp10 = tmp9 >= tmp5
    tmp11 = 4.0
    tmp12 = tmp9 < tmp11
    tmp14 = tmp13 * tmp1
    tmp15 = tmp14 + tmp3
    tmp16 = triton_helpers.maximum(tmp15, tmp5)
    tmp17 = triton_helpers.minimum(tmp16, tmp7)
    tmp18 = libdevice.floor(tmp17)
    tmp19 = tmp18 >= tmp5
    tmp20 = tmp18 < tmp11
    tmp21 = tmp19 & tmp20
    tmp22 = tmp12 & tmp21
    tmp23 = tmp10 & tmp22
    tmp24 = 1.0
    tmp25 = tmp9 + tmp24
    tmp26 = tmp25 - tmp8
    tmp27 = tmp18 + tmp24
    tmp28 = tmp27 - tmp17
    tmp29 = tmp26 * tmp28
    tmp30 = tl.where(tmp23, tmp29, tmp5)
    tmp31 = tmp25 >= tmp5
    tmp32 = tmp25 < tmp11
    tmp33 = tmp32 & tmp21
    tmp34 = tmp31 & tmp33
    tmp35 = tmp8 - tmp9
    tmp36 = tmp35 * tmp28
    tmp37 = tl.where(tmp34, tmp36, tmp5)
    tmp38 = tmp27 >= tmp5
    tmp39 = tmp27 < tmp11
    tmp40 = tmp38 & tmp39
    tmp41 = tmp12 & tmp40
    tmp42 = tmp10 & tmp41
    tmp43 = tmp17 - tmp18
    tmp44 = tmp26 * tmp43
    tmp45 = tl.where(tmp42, tmp44, tmp5)
    tmp46 = tmp18.to(tl.int64)
    tmp47 = tl.full([1], 0, tl.int64)
    tmp48 = tl.where(tmp23, tmp46, tmp47)
    tmp49 = tl.full([XBLOCK], 4, tl.int32)
    tmp50 = tmp48 + tmp49
    tmp51 = tmp48 < 0
    tmp52 = tl.where(tmp51, tmp50, tmp48)
    tl.device_assert(((0 <= tmp52) & (tmp52 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp52 < 4")
    tmp54 = tmp9.to(tl.int64)
    tmp55 = tl.where(tmp23, tmp54, tmp47)
    tmp56 = tmp55 + tmp49
    tmp57 = tmp55 < 0
    tmp58 = tl.where(tmp57, tmp56, tmp55)
    tl.device_assert(((0 <= tmp58) & (tmp58 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp58 < 4")
    tmp60 = tl.load(in_ptr1 + (tmp58 + 4*tmp52 + 16*x4), xmask, eviction_policy='evict_last')
    tmp61 = tl.where(tmp34, tmp46, tmp47)
    tmp62 = tmp61 + tmp49
    tmp63 = tmp61 < 0
    tmp64 = tl.where(tmp63, tmp62, tmp61)
    tl.device_assert(((0 <= tmp64) & (tmp64 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp64 < 4")
    tmp66 = tmp25.to(tl.int64)
    tmp67 = tl.where(tmp34, tmp66, tmp47)
    tmp68 = tmp67 + tmp49
    tmp69 = tmp67 < 0
    tmp70 = tl.where(tmp69, tmp68, tmp67)
    tl.device_assert(((0 <= tmp70) & (tmp70 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp70 < 4")
    tmp72 = tl.load(in_ptr1 + (tmp70 + 4*tmp64 + 16*x4), xmask, eviction_policy='evict_last')
    tmp73 = tmp27.to(tl.int64)
    tmp74 = tl.where(tmp42, tmp73, tmp47)
    tmp75 = tmp74 + tmp49
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tl.device_assert(((0 <= tmp77) & (tmp77 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp77 < 4")
    tmp79 = tl.where(tmp42, tmp54, tmp47)
    tmp80 = tmp79 + tmp49
    tmp81 = tmp79 < 0
    tmp82 = tl.where(tmp81, tmp80, tmp79)
    tl.device_assert(((0 <= tmp82) & (tmp82 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp82 < 4")
    tmp84 = tl.load(in_ptr1 + (tmp82 + 4*tmp77 + 16*x4), xmask, eviction_policy='evict_last')
    tmp85 = tmp32 & tmp40
    tmp86 = tmp31 & tmp85
    tmp87 = tl.where(tmp86, tmp73, tmp47)
    tmp88 = tmp87 + tmp49
    tmp89 = tmp87 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp87)
    tl.device_assert(((0 <= tmp90) & (tmp90 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp90 < 4")
    tmp92 = tl.where(tmp86, tmp66, tmp47)
    tmp93 = tmp92 + tmp49
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tl.device_assert(((0 <= tmp95) & (tmp95 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp95 < 4")
    tmp97 = tl.load(in_ptr1 + (tmp95 + 4*tmp90 + 16*x4), xmask, eviction_policy='evict_last')
    tmp98 = tmp35 * tmp43
    tmp99 = tl.where(tmp86, tmp98, tmp5)
    tmp100 = tmp60 * tmp30
    tmp101 = tmp72 * tmp37
    tmp102 = tmp100 + tmp101
    tmp103 = tmp84 * tmp45
    tmp104 = tmp102 + tmp103
    tmp105 = tmp97 * tmp99
    tmp106 = tmp104 + tmp105
    tl.store(in_out_ptr0 + (x3), tmp106, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 2, 4, 4), (32, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 2, 4, 4), (32, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 2), (32, 8, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_locs_2], Original ATen: [aten.index]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_0.run(arg0_1, arg1_1, buf0, 128, grid=grid(128), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf9 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [grid_sample], Original ATen: [aten.grid_sampler_2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_1.run(buf9, buf0, arg2_1, 256, grid=grid(256), stream=stream0)
        del arg2_1
        del buf0
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
