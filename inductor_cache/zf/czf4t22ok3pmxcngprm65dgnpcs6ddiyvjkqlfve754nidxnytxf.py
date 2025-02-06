# AOT ID: ['0_inference']
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


# kernel path: inductor_cache/sp/cspueeaughrekodd5iceuclbouwwe4ndtk6youyidhw3t2vvgp25.py
# Topologically Sorted Source Nodes: [base_grid_1, shift_1, grid], Original ATen: [aten.repeat, aten.mul, aten.add]
# Source node to ATen node mapping:
#   base_grid_1 => repeat_1
#   grid => add_1
#   shift_1 => mul_2
# Graph fragment:
#   %repeat_1 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze_2, [4, 1, 1, 1]), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%randint, 0.16666666666666666), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%repeat_1, %mul_2), kwargs = {})
triton_poi_fused_add_mul_repeat_0 = async_compile.triton('triton_poi_fused_add_mul_repeat_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_repeat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_repeat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 4)
    x2 = ((xindex // 8) % 4)
    x3 = xindex // 32
    x7 = xindex
    tmp41 = tl.load(in_ptr0 + (x0 + 2*x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 6.0
    tmp8 = tmp6 < tmp7
    tmp9 = 0.16666666666666666
    tmp10 = tmp6 * tmp9
    tmp11 = -0.9166666666666666
    tmp12 = tmp10 + tmp11
    tmp13 = 11 + ((-1)*x1)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp9
    tmp16 = 0.9166666666666666
    tmp17 = tmp16 - tmp15
    tmp18 = tl.where(tmp8, tmp12, tmp17)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp4, tmp18, tmp19)
    tmp21 = tmp0 >= tmp3
    tmp22 = tl.full([1], 2, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = x2
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 6.0
    tmp27 = tmp25 < tmp26
    tmp28 = 0.16666666666666666
    tmp29 = tmp25 * tmp28
    tmp30 = -0.9166666666666666
    tmp31 = tmp29 + tmp30
    tmp32 = 11 + ((-1)*x2)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp28
    tmp35 = 0.9166666666666666
    tmp36 = tmp35 - tmp34
    tmp37 = tl.where(tmp27, tmp31, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp21, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp20, tmp39)
    tmp42 = 0.16666666666666666
    tmp43 = tmp41 * tmp42
    tmp44 = tmp40 + tmp43
    tl.store(out_ptr0 + (x7), tmp44, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j3/cj3pziczbs44jnzqfizqx6rtn3pbfvkuwrposmc47oe4tdl7g32k.py
# Topologically Sorted Source Nodes: [x_1, grid_sample], Original ATen: [aten.replication_pad2d, aten.grid_sampler_2d]
# Source node to ATen node mapping:
#   grid_sample => add_2, add_3, add_4, add_5, add_6, add_7, add_8, floor, floor_1, full_default_11, full_default_2, full_default_5, full_default_8, ge, ge_1, ge_2, ge_3, ge_4, ge_5, ge_6, ge_7, index, index_1, index_2, index_3, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_3, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt_1, lt_2, lt_3, lt_4, lt_5, lt_6, lt_7, lt_8, mul_10, mul_11, mul_12, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9, where_12, where_3, where_6, where_9
#   x_1 => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=4] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %clamp_max_1]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, 6.0), kwargs = {})
#   %add_2 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 5.5), kwargs = {})
#   %floor : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_2,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 12), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, 6.0), kwargs = {})
#   %add_3 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 5.5), kwargs = {})
#   %floor_1 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_3,), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 12), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_1, %lt_2), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_1, %logical_and), kwargs = {})
#   %logical_and_2 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %logical_and_1), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%_unsafe_index_1, [%view_1, %view_2, %where_2, %where_1]), kwargs = {})
#   %add_4 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor, 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %add_2), kwargs = {})
#   %add_5 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_1, 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_3), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %sub_3), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %mul_5, %full_default_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %where_3), kwargs = {})
#   %ge_2 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_4, 0), kwargs = {})
#   %lt_3 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_4, 12), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 12), kwargs = {})
#   %logical_and_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_3, %lt_4), kwargs = {})
#   %logical_and_4 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_3, %logical_and_3), kwargs = {})
#   %logical_and_5 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_2, %logical_and_4), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%_unsafe_index_1, [%view_1, %view_2, %where_5, %where_4]), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %floor), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_3), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %sub_5), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_5, %mul_6, %full_default_5), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_1, %where_6), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_10), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt_5 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 12), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_5, 0), kwargs = {})
#   %lt_6 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_5, 12), kwargs = {})
#   %logical_and_6 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_5, %lt_6), kwargs = {})
#   %logical_and_7 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_5, %logical_and_6), kwargs = {})
#   %logical_and_8 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_4, %logical_and_7), kwargs = {})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%_unsafe_index_1, [%view_1, %view_2, %where_8, %where_7]), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %add_2), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %floor_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %sub_7), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_8, %mul_7, %full_default_8), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_2, %where_9), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mul_11), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_4, 0), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_4, 12), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_5, 0), kwargs = {})
#   %lt_8 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_5, 12), kwargs = {})
#   %logical_and_9 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_7, %lt_8), kwargs = {})
#   %logical_and_10 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_7, %logical_and_9), kwargs = {})
#   %logical_and_11 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_6, %logical_and_10), kwargs = {})
#   %index_3 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%_unsafe_index_1, [%view_1, %view_2, %where_11, %where_10]), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %floor), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %floor_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %sub_9), kwargs = {})
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_11, %mul_8, %full_default_11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_3, %where_12), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %mul_12), kwargs = {})
triton_poi_fused_grid_sampler_2d_replication_pad2d_1 = async_compile.triton('triton_poi_fused_grid_sampler_2d_replication_pad2d_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_replication_pad2d_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_replication_pad2d_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex // 16
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp1 = 6.0
    tmp2 = tmp0 * tmp1
    tmp3 = 5.5
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.floor(tmp4)
    tmp6 = 0.0
    tmp7 = tmp5 >= tmp6
    tmp8 = 12.0
    tmp9 = tmp5 < tmp8
    tmp11 = tmp10 * tmp1
    tmp12 = tmp11 + tmp3
    tmp13 = libdevice.floor(tmp12)
    tmp14 = tmp13 >= tmp6
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp9 & tmp16
    tmp18 = tmp7 & tmp17
    tmp19 = tmp13.to(tl.int64)
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.full([XBLOCK], 12, tl.int32)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp21 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp21)
    tl.device_assert(((0 <= tmp25) & (tmp25 < 12)) | ~(xmask), "index out of bounds: 0 <= tmp25 < 12")
    tmp27 = tmp5.to(tl.int64)
    tmp28 = tl.where(tmp18, tmp27, tmp20)
    tmp29 = tmp28 + tmp22
    tmp30 = tmp28 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp28)
    tl.device_assert(((0 <= tmp31) & (tmp31 < 12)) | ~(xmask), "index out of bounds: 0 <= tmp31 < 12")
    tmp33 = tl.load(in_ptr1 + (4*((3) * ((3) <= (((0) * ((0) >= ((-4) + tmp25)) + ((-4) + tmp25) * (((-4) + tmp25) > (0))))) + (((0) * ((0) >= ((-4) + tmp25)) + ((-4) + tmp25) * (((-4) + tmp25) > (0)))) * ((((0) * ((0) >= ((-4) + tmp25)) + ((-4) + tmp25) * (((-4) + tmp25) > (0)))) < (3))) + 16*x3 + ((3) * ((3) <= (((0) * ((0) >= ((-4) + tmp31)) + ((-4) + tmp31) * (((-4) + tmp31) > (0))))) + (((0) * ((0) >= ((-4) + tmp31)) + ((-4) + tmp31) * (((-4) + tmp31) > (0)))) * ((((0) * ((0) >= ((-4) + tmp31)) + ((-4) + tmp31) * (((-4) + tmp31) > (0)))) < (3)))), xmask, eviction_policy='evict_last')
    tmp34 = 1.0
    tmp35 = tmp5 + tmp34
    tmp36 = tmp35 >= tmp6
    tmp37 = tmp35 < tmp8
    tmp38 = tmp37 & tmp16
    tmp39 = tmp36 & tmp38
    tmp40 = tl.where(tmp39, tmp19, tmp20)
    tmp41 = tmp40 + tmp22
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tl.device_assert(((0 <= tmp43) & (tmp43 < 12)) | ~(xmask), "index out of bounds: 0 <= tmp43 < 12")
    tmp45 = tmp35.to(tl.int64)
    tmp46 = tl.where(tmp39, tmp45, tmp20)
    tmp47 = tmp46 + tmp22
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert(((0 <= tmp49) & (tmp49 < 12)) | ~(xmask), "index out of bounds: 0 <= tmp49 < 12")
    tmp51 = tl.load(in_ptr1 + (4*((3) * ((3) <= (((0) * ((0) >= ((-4) + tmp43)) + ((-4) + tmp43) * (((-4) + tmp43) > (0))))) + (((0) * ((0) >= ((-4) + tmp43)) + ((-4) + tmp43) * (((-4) + tmp43) > (0)))) * ((((0) * ((0) >= ((-4) + tmp43)) + ((-4) + tmp43) * (((-4) + tmp43) > (0)))) < (3))) + 16*x3 + ((3) * ((3) <= (((0) * ((0) >= ((-4) + tmp49)) + ((-4) + tmp49) * (((-4) + tmp49) > (0))))) + (((0) * ((0) >= ((-4) + tmp49)) + ((-4) + tmp49) * (((-4) + tmp49) > (0)))) * ((((0) * ((0) >= ((-4) + tmp49)) + ((-4) + tmp49) * (((-4) + tmp49) > (0)))) < (3)))), xmask, eviction_policy='evict_last')
    tmp52 = tmp35 - tmp4
    tmp53 = tmp13 + tmp34
    tmp54 = tmp53 - tmp12
    tmp55 = tmp52 * tmp54
    tmp56 = tl.where(tmp18, tmp55, tmp6)
    tmp57 = tmp33 * tmp56
    tmp58 = tmp4 - tmp5
    tmp59 = tmp58 * tmp54
    tmp60 = tl.where(tmp39, tmp59, tmp6)
    tmp61 = tmp51 * tmp60
    tmp62 = tmp57 + tmp61
    tmp63 = tmp53 >= tmp6
    tmp64 = tmp53 < tmp8
    tmp65 = tmp63 & tmp64
    tmp66 = tmp9 & tmp65
    tmp67 = tmp7 & tmp66
    tmp68 = tmp53.to(tl.int64)
    tmp69 = tl.where(tmp67, tmp68, tmp20)
    tmp70 = tmp69 + tmp22
    tmp71 = tmp69 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp69)
    tl.device_assert(((0 <= tmp72) & (tmp72 < 12)) | ~(xmask), "index out of bounds: 0 <= tmp72 < 12")
    tmp74 = tl.where(tmp67, tmp27, tmp20)
    tmp75 = tmp74 + tmp22
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tl.device_assert(((0 <= tmp77) & (tmp77 < 12)) | ~(xmask), "index out of bounds: 0 <= tmp77 < 12")
    tmp79 = tl.load(in_ptr1 + (4*((3) * ((3) <= (((0) * ((0) >= ((-4) + tmp72)) + ((-4) + tmp72) * (((-4) + tmp72) > (0))))) + (((0) * ((0) >= ((-4) + tmp72)) + ((-4) + tmp72) * (((-4) + tmp72) > (0)))) * ((((0) * ((0) >= ((-4) + tmp72)) + ((-4) + tmp72) * (((-4) + tmp72) > (0)))) < (3))) + 16*x3 + ((3) * ((3) <= (((0) * ((0) >= ((-4) + tmp77)) + ((-4) + tmp77) * (((-4) + tmp77) > (0))))) + (((0) * ((0) >= ((-4) + tmp77)) + ((-4) + tmp77) * (((-4) + tmp77) > (0)))) * ((((0) * ((0) >= ((-4) + tmp77)) + ((-4) + tmp77) * (((-4) + tmp77) > (0)))) < (3)))), xmask, eviction_policy='evict_last')
    tmp80 = tmp12 - tmp13
    tmp81 = tmp52 * tmp80
    tmp82 = tl.where(tmp67, tmp81, tmp6)
    tmp83 = tmp79 * tmp82
    tmp84 = tmp62 + tmp83
    tmp85 = tmp37 & tmp65
    tmp86 = tmp36 & tmp85
    tmp87 = tl.where(tmp86, tmp68, tmp20)
    tmp88 = tmp87 + tmp22
    tmp89 = tmp87 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp87)
    tl.device_assert(((0 <= tmp90) & (tmp90 < 12)) | ~(xmask), "index out of bounds: 0 <= tmp90 < 12")
    tmp92 = tl.where(tmp86, tmp45, tmp20)
    tmp93 = tmp92 + tmp22
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tl.device_assert(((0 <= tmp95) & (tmp95 < 12)) | ~(xmask), "index out of bounds: 0 <= tmp95 < 12")
    tmp97 = tl.load(in_ptr1 + (4*((3) * ((3) <= (((0) * ((0) >= ((-4) + tmp90)) + ((-4) + tmp90) * (((-4) + tmp90) > (0))))) + (((0) * ((0) >= ((-4) + tmp90)) + ((-4) + tmp90) * (((-4) + tmp90) > (0)))) * ((((0) * ((0) >= ((-4) + tmp90)) + ((-4) + tmp90) * (((-4) + tmp90) > (0)))) < (3))) + 16*x3 + ((3) * ((3) <= (((0) * ((0) >= ((-4) + tmp95)) + ((-4) + tmp95) * (((-4) + tmp95) > (0))))) + (((0) * ((0) >= ((-4) + tmp95)) + ((-4) + tmp95) * (((-4) + tmp95) > (0)))) * ((((0) * ((0) >= ((-4) + tmp95)) + ((-4) + tmp95) * (((-4) + tmp95) > (0)))) < (3)))), xmask, eviction_policy='evict_last')
    tmp98 = tmp58 * tmp80
    tmp99 = tl.where(tmp86, tmp98, tmp6)
    tmp100 = tmp97 * tmp99
    tmp101 = tmp84 + tmp100
    tl.store(in_out_ptr0 + (x4), tmp101, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [shift], Original ATen: [aten.randint]
        buf1 = torch.ops.aten.randint.low(0, 9, [4, 1, 1, 2], dtype=torch.float32, device=device(type='cuda', index=0), pin_memory=False)
        buf2 = buf1
        del buf1
        buf3 = empty_strided_cuda((4, 4, 4, 2), (32, 8, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [base_grid_1, shift_1, grid], Original ATen: [aten.repeat, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_repeat_0.run(buf2, buf3, 128, grid=grid(128), stream=stream0)
        del buf2
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf6 = buf4; del buf4  # reuse
        buf8 = buf6; del buf6  # reuse
        buf10 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_1, grid_sample], Original ATen: [aten.replication_pad2d, aten.grid_sampler_2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_replication_pad2d_1.run(buf10, buf3, arg0_1, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del buf3
    return (buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
