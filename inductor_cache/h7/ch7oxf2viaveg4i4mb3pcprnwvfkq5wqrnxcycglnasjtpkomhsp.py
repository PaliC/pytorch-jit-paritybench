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


# kernel path: inductor_cache/h3/ch3rvub25zkmn2pmzmatypy4tjv75uhi5umyulrjc6mxwvymwe3w.py
# Topologically Sorted Source Nodes: [cat, vgrid, sub, setitem, clone, mul, truediv, sub_1, setitem_1], Original ATen: [aten.cat, aten._to_copy, aten.sub, aten.copy, aten.clone, aten.mul, aten.div]
# Source node to ATen node mapping:
#   cat => cat
#   clone => clone
#   mul => mul
#   setitem => copy
#   setitem_1 => copy_1
#   sub => sub
#   sub_1 => sub_1
#   truediv => div
#   vgrid => convert_element_type
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_4, %repeat_5], 1), kwargs = {})
#   %convert_element_type : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_10, %view), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_14, %sub), kwargs = {})
#   %slice_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%convert_element_type, %copy, 1, 0, 1), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone, 2.0), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, 1.0), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_3, %sub_1), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default, %copy_1, 1, 0), kwargs = {})
triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_0 = async_compile.triton('triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 2)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x3 = xindex // 32
    x4 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp6 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = x0
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tmp6 >= tmp8
    tmp15 = tl.full([1], 2, tl.int64)
    tmp16 = tmp6 < tmp15
    tmp17 = tmp14 & tmp5
    tmp18 = x1
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tl.where(tmp9, tmp13, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.load(in_ptr0 + (x4 + 16*x3), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 - tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp5, tmp24, tmp25)
    tmp27 = tmp3 >= tmp3
    tmp28 = x0
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp5, tmp28, tmp29)
    tmp31 = tmp3 >= tmp4
    tmp32 = tl.full([1], 2, tl.int64)
    tmp33 = tmp3 < tmp32
    tmp34 = x1
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = tl.where(tmp5, tmp30, tmp36)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.where(tmp5, tmp26, tmp38)
    tmp40 = 2.0
    tmp41 = tmp39 * tmp40
    tmp42 = 0.3333333333333333
    tmp43 = tmp41 * tmp42
    tmp44 = 1.0
    tmp45 = tmp43 - tmp44
    tmp46 = tmp0 < tmp4
    tmp47 = x2
    tmp48 = tl.full([1], 0, tl.int64)
    tmp49 = tmp47 >= tmp48
    tmp50 = tl.full([1], 1, tl.int64)
    tmp51 = tmp47 < tmp50
    tmp52 = tmp51 & tmp46
    tmp53 = x0
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp47 >= tmp50
    tmp57 = tl.full([1], 2, tl.int64)
    tmp58 = tmp47 < tmp57
    tmp59 = tmp56 & tmp46
    tmp60 = x1
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = tl.where(tmp51, tmp55, tmp62)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tl.load(in_ptr0 + (x4 + 16*x3), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 - tmp65
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp46, tmp66, tmp67)
    tmp69 = tmp0 >= tmp3
    tmp70 = x0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp46, tmp70, tmp71)
    tmp73 = tmp0 >= tmp4
    tmp74 = tmp0 < tmp32
    tmp75 = x1
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp73, tmp75, tmp76)
    tmp78 = tl.where(tmp46, tmp72, tmp77)
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tl.where(tmp46, tmp68, tmp79)
    tmp81 = tl.where(tmp2, tmp45, tmp80)
    tl.store(out_ptr0 + (x5), tmp81, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hn/chn2tprw227k5tpgo2r7smih3cuvujafr4zubzmat7kkgetpyanf.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.grid_sampler_2d]
# Source node to ATen node mapping:
#   output => add, add_1, add_2, add_3, convert_element_type_1, convert_element_type_2, convert_element_type_8, floor, floor_1, full_default, full_default_1, full_default_10, full_default_11, full_default_2, full_default_5, full_default_8, ge, ge_1, ge_2, ge_3, ge_4, ge_5, ge_6, ge_7, index_1, index_2, index_3, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_3, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt, lt_1, lt_2, lt_3, lt_4, lt_5, lt_6, lt_7, mul_10, mul_11, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_9, sub_10, sub_3, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9, where, where_1, where_10, where_11, where_2, where_5, where_8
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, 2.0), kwargs = {})
#   %add : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 1.5), kwargs = {})
#   %floor : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, 2.0), kwargs = {})
#   %add_1 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 1.5), kwargs = {})
#   %floor_1 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_1,), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_1, %lt_1), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt, %logical_and), kwargs = {})
#   %logical_and_2 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %logical_and_1), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_1, torch.int64), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %convert_element_type_2, %full_default_1), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %convert_element_type_1, %full_default), kwargs = {})
#   %add_2 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor, 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %add), kwargs = {})
#   %add_3 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_1, 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %add_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %sub_4), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %mul_4, %full_default_2), kwargs = {})
#   %ge_2 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_2, 0), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_2, 4), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_3 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_3, %lt_3), kwargs = {})
#   %logical_and_4 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_2, %logical_and_3), kwargs = {})
#   %logical_and_5 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_2, %logical_and_4), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_2, [%view_8, %view_9, %where_4, %where_3]), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %floor), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %add_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %sub_6), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_5, %mul_5, %full_default_5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_1, %where_5), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_3, 0), kwargs = {})
#   %lt_5 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_3, 4), kwargs = {})
#   %logical_and_6 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_5, %lt_5), kwargs = {})
#   %logical_and_7 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_4, %logical_and_6), kwargs = {})
#   %logical_and_8 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_4, %logical_and_7), kwargs = {})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_2, [%view_8, %view_9, %where_7, %where_6]), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %add), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %floor_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %sub_8), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_8, %mul_6, %full_default_8), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_2, %where_8), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_2, 0), kwargs = {})
#   %lt_6 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_2, 4), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_3, 0), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_3, 4), kwargs = {})
#   %logical_and_9 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_7, %lt_7), kwargs = {})
#   %logical_and_10 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_6, %logical_and_9), kwargs = {})
#   %logical_and_11 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_6, %logical_and_10), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.int64), kwargs = {})
#   %full_default_10 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_11, %convert_element_type_8, %full_default_10), kwargs = {})
#   %index_3 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_2, [%view_8, %view_9, %where_10, %where_9]), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %floor), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %floor_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %sub_10), kwargs = {})
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_11, %mul_7, %full_default_11), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_3, %where_11), kwargs = {})
triton_poi_fused_grid_sampler_2d_1 = async_compile.triton('triton_poi_fused_grid_sampler_2d_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_1(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.3333333333333333
    tmp7 = tmp5 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 - tmp8
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp11 * tmp4
    tmp13 = 1.5
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = 0.0
    tmp17 = tmp15 >= tmp16
    tmp18 = 4.0
    tmp19 = tmp15 < tmp18
    tmp20 = tmp1 == tmp1
    tmp21 = tl.where(tmp20, tmp9, tmp3)
    tmp22 = tmp21 * tmp4
    tmp23 = tmp22 + tmp13
    tmp24 = libdevice.floor(tmp23)
    tmp25 = tmp24 >= tmp16
    tmp26 = tmp24 < tmp18
    tmp27 = tmp25 & tmp26
    tmp28 = tmp19 & tmp27
    tmp29 = tmp17 & tmp28
    tmp30 = tmp24.to(tl.int64)
    tmp31 = tl.full([1], 0, tl.int64)
    tmp32 = tl.where(tmp29, tmp30, tmp31)
    tmp33 = tmp15.to(tl.int64)
    tmp34 = tl.where(tmp29, tmp33, tmp31)
    tmp35 = tmp15 + tmp8
    tmp36 = tmp35 - tmp14
    tmp37 = tmp24 + tmp8
    tmp38 = tmp37 - tmp23
    tmp39 = tmp36 * tmp38
    tmp40 = tl.where(tmp29, tmp39, tmp16)
    tmp41 = tmp35 >= tmp16
    tmp42 = tmp35 < tmp18
    tmp43 = tmp42 & tmp27
    tmp44 = tmp41 & tmp43
    tmp45 = tl.where(tmp44, tmp30, tmp31)
    tmp46 = tl.full([XBLOCK], 4, tl.int32)
    tmp47 = tmp45 + tmp46
    tmp48 = tmp45 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp45)
    tl.device_assert(((0 <= tmp49) & (tmp49 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp49 < 4")
    tmp51 = tmp35.to(tl.int64)
    tmp52 = tl.where(tmp44, tmp51, tmp31)
    tmp53 = tmp52 + tmp46
    tmp54 = tmp52 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp52)
    tl.device_assert(((0 <= tmp55) & (tmp55 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp55 < 4")
    tmp57 = tl.load(in_ptr1 + (tmp55 + 4*tmp49 + 16*x1 + 64*(x2 // 4)), xmask, eviction_policy='evict_last')
    tmp58 = tmp14 - tmp15
    tmp59 = tmp58 * tmp38
    tmp60 = tl.where(tmp44, tmp59, tmp16)
    tmp61 = tmp57 * tmp60
    tmp62 = tmp37 >= tmp16
    tmp63 = tmp37 < tmp18
    tmp64 = tmp62 & tmp63
    tmp65 = tmp19 & tmp64
    tmp66 = tmp17 & tmp65
    tmp67 = tmp37.to(tl.int64)
    tmp68 = tl.where(tmp66, tmp67, tmp31)
    tmp69 = tmp68 + tmp46
    tmp70 = tmp68 < 0
    tmp71 = tl.where(tmp70, tmp69, tmp68)
    tl.device_assert(((0 <= tmp71) & (tmp71 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp71 < 4")
    tmp73 = tl.where(tmp66, tmp33, tmp31)
    tmp74 = tmp73 + tmp46
    tmp75 = tmp73 < 0
    tmp76 = tl.where(tmp75, tmp74, tmp73)
    tl.device_assert(((0 <= tmp76) & (tmp76 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp76 < 4")
    tmp78 = tl.load(in_ptr1 + (tmp76 + 4*tmp71 + 16*x1 + 64*(x2 // 4)), xmask, eviction_policy='evict_last')
    tmp79 = tmp23 - tmp24
    tmp80 = tmp36 * tmp79
    tmp81 = tl.where(tmp66, tmp80, tmp16)
    tmp82 = tmp78 * tmp81
    tmp83 = tmp42 & tmp64
    tmp84 = tmp41 & tmp83
    tmp85 = tmp58 * tmp79
    tmp86 = tl.where(tmp84, tmp85, tmp16)
    tmp87 = tl.where(tmp84, tmp67, tmp31)
    tmp88 = tmp87 + tmp46
    tmp89 = tmp87 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp87)
    tl.device_assert(((0 <= tmp90) & (tmp90 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp90 < 4")
    tmp92 = tl.where(tmp84, tmp51, tmp31)
    tmp93 = tmp92 + tmp46
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tl.device_assert(((0 <= tmp95) & (tmp95 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp95 < 4")
    tmp97 = tl.load(in_ptr1 + (tmp95 + 4*tmp90 + 16*x1 + 64*(x2 // 4)), xmask, eviction_policy='evict_last')
    tmp98 = tmp97 * tmp86
    tl.store(out_ptr0 + (x3), tmp32, xmask)
    tl.store(out_ptr1 + (x3), tmp34, xmask)
    tl.store(out_ptr2 + (x3), tmp40, xmask)
    tl.store(in_out_ptr0 + (x3), tmp61, xmask)
    tl.store(in_out_ptr1 + (x3), tmp82, xmask)
    tl.store(in_out_ptr2 + (x3), tmp98, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qg/cqgutobqtjpamlfkyjlcd4hlp3ym47b4g3lzgkixrnibrg6hcevz.py
# Topologically Sorted Source Nodes: [output, sub_3, norm], Original ATen: [aten.grid_sampler_2d, aten.sub, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   norm => abs_1, pow_2, sum_1
#   output => add_4, add_5, add_6, index, mul_8
#   sub_3 => sub_11
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_2, [%view_8, %view_9, %where_1, %where]), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %where_2), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_10), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_11), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %add_6), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_11,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 1.0), kwargs = {})
triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_2 = async_compile.triton('triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 28, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*(x1 // 4)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp7 = tl.load(in_ptr2 + (x0 + 64*x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x0 + 64*x1), xmask)
    tmp15 = tl.load(in_ptr5 + (x0 + 64*x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x0 + 64*x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x0 + 64*x1), xmask)
    tmp23 = tl.load(in_ptr0 + (16 + x0 + 64*(x1 // 4)), xmask)
    tmp24 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp29 = tl.load(in_ptr2 + (16 + x0 + 64*x1), xmask)
    tmp35 = tl.load(in_ptr4 + (16 + x0 + 64*x1), xmask)
    tmp37 = tl.load(in_ptr5 + (16 + x0 + 64*x1), xmask)
    tmp39 = tl.load(in_ptr6 + (16 + x0 + 64*x1), xmask)
    tmp41 = tl.load(in_ptr7 + (16 + x0 + 64*x1), xmask)
    tmp46 = tl.load(in_ptr0 + (32 + x0 + 64*(x1 // 4)), xmask)
    tmp47 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp52 = tl.load(in_ptr2 + (32 + x0 + 64*x1), xmask)
    tmp58 = tl.load(in_ptr4 + (32 + x0 + 64*x1), xmask)
    tmp60 = tl.load(in_ptr5 + (32 + x0 + 64*x1), xmask)
    tmp62 = tl.load(in_ptr6 + (32 + x0 + 64*x1), xmask)
    tmp64 = tl.load(in_ptr7 + (32 + x0 + 64*x1), xmask)
    tmp69 = tl.load(in_ptr0 + (48 + x0 + 64*(x1 // 4)), xmask)
    tmp70 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp75 = tl.load(in_ptr2 + (48 + x0 + 64*x1), xmask)
    tmp81 = tl.load(in_ptr4 + (48 + x0 + 64*x1), xmask)
    tmp83 = tl.load(in_ptr5 + (48 + x0 + 64*x1), xmask)
    tmp85 = tl.load(in_ptr6 + (48 + x0 + 64*x1), xmask)
    tmp87 = tl.load(in_ptr7 + (48 + x0 + 64*x1), xmask)
    tmp2 = tl.full([XBLOCK], 4, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 4")
    tmp8 = tmp7 + tmp2
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 4")
    tmp12 = tl.load(in_ptr3 + (tmp10 + 4*tmp5 + 64*(x1 // 4)), xmask, eviction_policy='evict_last')
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp21 = tmp0 - tmp20
    tmp22 = tl_math.abs(tmp21)
    tmp25 = tmp24 + tmp2
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp27 < 4")
    tmp30 = tmp29 + tmp2
    tmp31 = tmp29 < 0
    tmp32 = tl.where(tmp31, tmp30, tmp29)
    tl.device_assert(((0 <= tmp32) & (tmp32 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp32 < 4")
    tmp34 = tl.load(in_ptr3 + (16 + tmp32 + 4*tmp27 + 64*(x1 // 4)), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp23 - tmp42
    tmp44 = tl_math.abs(tmp43)
    tmp45 = tmp22 + tmp44
    tmp48 = tmp47 + tmp2
    tmp49 = tmp47 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp47)
    tl.device_assert(((0 <= tmp50) & (tmp50 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp50 < 4")
    tmp53 = tmp52 + tmp2
    tmp54 = tmp52 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp52)
    tl.device_assert(((0 <= tmp55) & (tmp55 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp55 < 4")
    tmp57 = tl.load(in_ptr3 + (32 + tmp55 + 4*tmp50 + 64*(x1 // 4)), xmask, eviction_policy='evict_last')
    tmp59 = tmp57 * tmp58
    tmp61 = tmp59 + tmp60
    tmp63 = tmp61 + tmp62
    tmp65 = tmp63 + tmp64
    tmp66 = tmp46 - tmp65
    tmp67 = tl_math.abs(tmp66)
    tmp68 = tmp45 + tmp67
    tmp71 = tmp70 + tmp2
    tmp72 = tmp70 < 0
    tmp73 = tl.where(tmp72, tmp71, tmp70)
    tl.device_assert(((0 <= tmp73) & (tmp73 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp73 < 4")
    tmp76 = tmp75 + tmp2
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tl.device_assert(((0 <= tmp78) & (tmp78 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp78 < 4")
    tmp80 = tl.load(in_ptr3 + (48 + tmp78 + 4*tmp73 + 64*(x1 // 4)), xmask, eviction_policy='evict_last')
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp86 = tmp84 + tmp85
    tmp88 = tmp86 + tmp87
    tmp89 = tmp69 - tmp88
    tmp90 = tl_math.abs(tmp89)
    tmp91 = tmp68 + tmp90
    tl.store(in_out_ptr0 + (x2), tmp91, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat, vgrid, sub, setitem, clone, mul, truediv, sub_1, setitem_1], Original ATen: [aten.cat, aten._to_copy, aten.sub, aten.copy, aten.clone, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_0.run(arg1_1, buf0, 512, grid=grid(512), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((16, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        buf2 = empty_strided_cuda((16, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        buf3 = empty_strided_cuda((16, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf5 = empty_strided_cuda((16, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf6 = buf5; del buf5  # reuse
        buf8 = empty_strided_cuda((16, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf9 = buf8; del buf8  # reuse
        buf12 = empty_strided_cuda((16, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.grid_sampler_2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_1.run(buf6, buf9, buf13, buf0, arg2_1, buf1, buf2, buf3, 1024, grid=grid(1024), stream=stream0)
        del buf0
        buf14 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [output, sub_3, norm], Original ATen: [aten.grid_sampler_2d, aten.sub, aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_2.run(buf15, arg0_1, buf1, buf2, arg2_1, buf3, buf6, buf9, buf13, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del arg2_1
        del buf1
        del buf13
        del buf2
        del buf3
        del buf6
        del buf9
    return (reinterpret_tensor(buf15, (4, 4, 4, 4), (64, 16, 4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
