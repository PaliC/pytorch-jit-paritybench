# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/hc/chceld3qnllw7figer5r6fck7agziow7l76xaannrp3ob7biu7ui.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.leaky_relu, aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_1 => gt, mul, where
#   input_2 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add, add_4, add_5, add_6, clamp_max_2, clamp_max_3, clamp_min, clamp_min_2, clamp_min_3, convert_element_type, convert_element_type_1, convert_element_type_3, iota, mul_1, mul_3, mul_4, mul_5, sub, sub_2, sub_3, sub_4, sub_5, sub_6
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%primals_1, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, 0.2), kwargs = {})
#   %where : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt, %primals_1, %mul), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 1.0), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min, torch.int64), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_3), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_4), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_5), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_sub_0 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_sub_0', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_sub_0', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_sub_0(in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = x0
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12 * tmp4
    tmp14 = tmp13 - tmp2
    tmp15 = triton_helpers.maximum(tmp14, tmp7)
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tl.full([1], 1, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1], 3, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tl.load(in_ptr0 + (tmp20 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp22 = tmp21 > tmp7
    tmp23 = 0.2
    tmp24 = tmp21 * tmp23
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp26 = tl.load(in_ptr0 + (tmp16 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp27 = tmp26 > tmp7
    tmp28 = tmp26 * tmp23
    tmp29 = tl.where(tmp27, tmp26, tmp28)
    tmp30 = tmp25 - tmp29
    tmp31 = tmp9 + tmp17
    tmp32 = triton_helpers.minimum(tmp31, tmp19)
    tmp33 = tl.load(in_ptr0 + (tmp20 + 4*tmp32 + 16*x2), xmask, eviction_policy='evict_last')
    tmp34 = tmp33 > tmp7
    tmp35 = tmp33 * tmp23
    tmp36 = tl.where(tmp34, tmp33, tmp35)
    tmp37 = tl.load(in_ptr0 + (tmp16 + 4*tmp32 + 16*x2), xmask, eviction_policy='evict_last')
    tmp38 = tmp37 > tmp7
    tmp39 = tmp37 * tmp23
    tmp40 = tl.where(tmp38, tmp37, tmp39)
    tmp41 = tmp36 - tmp40
    tmp42 = tmp16.to(tl.float32)
    tmp43 = tmp15 - tmp42
    tmp44 = triton_helpers.maximum(tmp43, tmp7)
    tmp45 = triton_helpers.minimum(tmp44, tmp4)
    tmp46 = tmp41 * tmp45
    tmp47 = tmp40 + tmp46
    tmp48 = tmp30 * tmp45
    tmp49 = tmp29 + tmp48
    tmp50 = tmp47 - tmp49
    tmp51 = tmp9.to(tl.float32)
    tmp52 = tmp8 - tmp51
    tmp53 = triton_helpers.maximum(tmp52, tmp7)
    tmp54 = triton_helpers.minimum(tmp53, tmp4)
    tmp55 = tmp50 * tmp54
    tmp56 = tmp49 + tmp55
    tl.store(in_out_ptr1 + (x4), tmp56, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ex/cexyiulzlunx75dc5r67o3mryye3zpamxabfpol7anfkmlhw6hmk.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   out => sum_1
# Graph fragment:
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_2, [1]), kwargs = {})
triton_poi_fused_sum_1 = async_compile.triton('triton_poi_fused_sum_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (x0 + 64*x1), tmp3 & xmask, other=0.0)
    tmp5 = tmp0 >= tmp2
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp0 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tl.load(in_ptr0 + (16 + x0 + 64*x1), tmp8 & xmask, other=0.0)
    tmp10 = tmp0 >= tmp6
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr0 + (32 + x0 + 64*x1), tmp13 & xmask, other=0.0)
    tmp15 = tmp0 >= tmp11
    tmp16 = tl.full([1], 4, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr0 + (48 + x0 + 64*x1), tmp15 & xmask, other=0.0)
    tmp19 = tl.where(tmp13, tmp14, tmp18)
    tmp20 = tl.where(tmp8, tmp9, tmp19)
    tmp21 = tl.where(tmp3, tmp4, tmp20)
    tmp22 = tmp2 >= tmp0
    tmp23 = tmp2 < tmp2
    tmp24 = tl.load(in_ptr0 + (x0 + 64*x1), tmp23 & xmask, other=0.0)
    tmp25 = tmp2 >= tmp2
    tmp26 = tmp2 < tmp6
    tmp27 = tmp25 & tmp26
    tmp28 = tl.load(in_ptr0 + (16 + x0 + 64*x1), tmp27 & xmask, other=0.0)
    tmp29 = tmp2 >= tmp6
    tmp30 = tmp2 < tmp11
    tmp31 = tmp29 & tmp30
    tmp32 = tl.load(in_ptr0 + (32 + x0 + 64*x1), tmp31 & xmask, other=0.0)
    tmp33 = tmp2 >= tmp11
    tmp34 = tmp2 < tmp16
    tmp35 = tl.load(in_ptr0 + (48 + x0 + 64*x1), tmp33 & xmask, other=0.0)
    tmp36 = tl.where(tmp31, tmp32, tmp35)
    tmp37 = tl.where(tmp27, tmp28, tmp36)
    tmp38 = tl.where(tmp23, tmp24, tmp37)
    tmp39 = tmp21 + tmp38
    tmp40 = tmp6 >= tmp0
    tmp41 = tmp6 < tmp2
    tmp42 = tl.load(in_ptr0 + (x0 + 64*x1), tmp41 & xmask, other=0.0)
    tmp43 = tmp6 >= tmp2
    tmp44 = tmp6 < tmp6
    tmp45 = tmp43 & tmp44
    tmp46 = tl.load(in_ptr0 + (16 + x0 + 64*x1), tmp45 & xmask, other=0.0)
    tmp47 = tmp6 >= tmp6
    tmp48 = tmp6 < tmp11
    tmp49 = tmp47 & tmp48
    tmp50 = tl.load(in_ptr0 + (32 + x0 + 64*x1), tmp49 & xmask, other=0.0)
    tmp51 = tmp6 >= tmp11
    tmp52 = tmp6 < tmp16
    tmp53 = tl.load(in_ptr0 + (48 + x0 + 64*x1), tmp51 & xmask, other=0.0)
    tmp54 = tl.where(tmp49, tmp50, tmp53)
    tmp55 = tl.where(tmp45, tmp46, tmp54)
    tmp56 = tl.where(tmp41, tmp42, tmp55)
    tmp57 = tmp39 + tmp56
    tmp58 = tmp11 >= tmp0
    tmp59 = tmp11 < tmp2
    tmp60 = tl.load(in_ptr0 + (x0 + 64*x1), tmp59 & xmask, other=0.0)
    tmp61 = tmp11 >= tmp2
    tmp62 = tmp11 < tmp6
    tmp63 = tmp61 & tmp62
    tmp64 = tl.load(in_ptr0 + (16 + x0 + 64*x1), tmp63 & xmask, other=0.0)
    tmp65 = tmp11 >= tmp6
    tmp66 = tmp11 < tmp11
    tmp67 = tmp65 & tmp66
    tmp68 = tl.load(in_ptr0 + (32 + x0 + 64*x1), tmp67 & xmask, other=0.0)
    tmp69 = tmp11 >= tmp11
    tmp70 = tmp11 < tmp16
    tmp71 = tl.load(in_ptr0 + (48 + x0 + 64*x1), tmp69 & xmask, other=0.0)
    tmp72 = tl.where(tmp67, tmp68, tmp71)
    tmp73 = tl.where(tmp63, tmp64, tmp72)
    tmp74 = tl.where(tmp59, tmp60, tmp73)
    tmp75 = tmp57 + tmp74
    tl.store(out_ptr0 + (x2), tmp75, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/op/coptfmyvjjjmsdezt45ud5zexgfnewa7hpnn45xcdogbwyog3mpx.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   input_4 => gt_1, mul_6, where_1
# Graph fragment:
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution, %mul_6), kwargs = {})
triton_poi_fused_leaky_relu_2 = async_compile.triton('triton_poi_fused_leaky_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_2(in_out_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 1, 4, 4), (16, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf4 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.leaky_relu, aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_sub_0.run(buf4, primals_1, 256, grid=grid(256), stream=stream0)
        del primals_1
        buf5 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_1.run(buf4, buf5, 64, grid=grid(64), stream=stream0)
        del buf4
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_2, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 4, 9, 9), (324, 81, 9, 1))
        buf7 = empty_strided_cuda((4, 4, 9, 9), (324, 81, 9, 1), torch.bool)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_2.run(buf8, buf7, 1296, grid=grid(1296), stream=stream0)
    return (buf8, primals_2, buf5, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 1, 4, 4), (16, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
