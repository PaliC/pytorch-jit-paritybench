# AOT ID: ['11_inference']
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


# kernel path: inductor_cache/im/cimqvpx3tgurczlfs6gkfpjrdvzygrbfsssjuaijdsvn7lwniylw.py
# Topologically Sorted Source Nodes: [x1, x0, x_ceil, y1, y0, y_ceil, y_floor, x_floor, sub, abs_1, sub_1, sub_2, abs_2, sub_3, mul_4, sub_4, abs_3, sub_5, sub_6, abs_4, sub_7, mul_5, sub_8, abs_5, sub_9, sub_10, abs_6, sub_11, mul_6, sub_12, abs_7, sub_13, sub_14, abs_8, sub_15, mul_7], Original ATen: [aten.floor, aten.add, aten.clamp, aten.sub, aten.abs, aten.rsub, aten.mul]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   abs_2 => abs_2
#   abs_3 => abs_3
#   abs_4 => abs_4
#   abs_5 => abs_5
#   abs_6 => abs_6
#   abs_7 => abs_7
#   abs_8 => abs_8
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   sub => sub
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sub_14 => sub_14
#   sub_15 => sub_15
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   sub_9 => sub_9
#   x0 => add_1
#   x1 => floor
#   x_ceil => clamp_max_2, clamp_min_2
#   x_floor => clamp_max, clamp_min
#   y0 => add_2
#   y1 => floor_1
#   y_ceil => clamp_max_3, clamp_min_3
#   y_floor => clamp_max_1, clamp_min_1
# Graph fragment:
#   %floor : [num_users=3] = call_function[target=torch.ops.aten.floor.default](args = (%view,), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor, 1), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0), kwargs = {})
#   %clamp_max_2 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 3), kwargs = {})
#   %floor_1 : [num_users=3] = call_function[target=torch.ops.aten.floor.default](args = (%view_1,), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_1, 1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_2, 0), kwargs = {})
#   %clamp_max_3 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 3), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%floor_1, 0), kwargs = {})
#   %clamp_max_1 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%floor, 0), kwargs = {})
#   %clamp_max : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 3), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %clamp_max_2), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %abs_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %clamp_max_3), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_2,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %abs_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %sub_3), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %clamp_max_2), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_4,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %abs_3), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %clamp_max_1), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_6,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %abs_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %sub_7), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %clamp_max), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_8,), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %abs_5), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %clamp_max_3), kwargs = {})
#   %abs_6 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_10,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %abs_6), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %sub_11), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %clamp_max), kwargs = {})
#   %abs_7 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_12,), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %abs_7), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %clamp_max_1), kwargs = {})
#   %abs_8 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_14,), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %abs_8), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %sub_15), kwargs = {})
triton_poi_fused_abs_add_clamp_floor_mul_rsub_sub_0 = async_compile.triton('triton_poi_fused_abs_add_clamp_floor_mul_rsub_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_floor_mul_rsub_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_floor_mul_rsub_sub_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tmp4 = tl.load(in_ptr0 + (x0 + 16*tmp3), xmask)
    tmp5 = tl.load(in_ptr1 + (x0 + 16*tmp3 + 32*x1), xmask)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.floor(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = 0.0
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = 3.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = tmp6 - tmp13
    tmp15 = tl_math.abs(tmp14)
    tmp16 = tmp8 - tmp15
    tmp17 = tmp1 < tmp1
    tmp18 = tl.where(tmp17, tmp1, tmp0)
    tmp19 = tl.load(in_ptr0 + (x0 + 16*tmp18), xmask)
    tmp20 = tl.load(in_ptr1 + (x0 + 16*tmp18 + 32*x1), xmask)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.floor(tmp21)
    tmp23 = tmp22 + tmp8
    tmp24 = triton_helpers.maximum(tmp23, tmp10)
    tmp25 = triton_helpers.minimum(tmp24, tmp12)
    tmp26 = tmp21 - tmp25
    tmp27 = tl_math.abs(tmp26)
    tmp28 = tmp8 - tmp27
    tmp29 = tmp16 * tmp28
    tmp30 = triton_helpers.maximum(tmp22, tmp10)
    tmp31 = triton_helpers.minimum(tmp30, tmp12)
    tmp32 = tmp21 - tmp31
    tmp33 = tl_math.abs(tmp32)
    tmp34 = tmp8 - tmp33
    tmp35 = tmp16 * tmp34
    tmp36 = triton_helpers.maximum(tmp7, tmp10)
    tmp37 = triton_helpers.minimum(tmp36, tmp12)
    tmp38 = tmp6 - tmp37
    tmp39 = tl_math.abs(tmp38)
    tmp40 = tmp8 - tmp39
    tmp41 = tmp40 * tmp28
    tmp42 = tmp40 * tmp34
    tl.store(out_ptr0 + (x0 + 64*x1), tmp29, xmask)
    tl.store(out_ptr1 + (x0 + 64*x1), tmp35, xmask)
    tl.store(out_ptr2 + (x0 + 64*x1), tmp41, xmask)
    tl.store(out_ptr3 + (x0 + 64*x1), tmp42, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wo/cwovmx7fnv5ijde2u4srkr5bctbgxe5vjdoag4jlzc4enqabkxvc.py
# Topologically Sorted Source Nodes: [corresponding_map, indices, scatter_add_], Original ATen: [aten._to_copy, aten.scatter_add]
# Source node to ATen node mapping:
#   corresponding_map => full_default
#   indices => convert_element_type_1
#   scatter_add_ => scatter_add
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 16], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.int64), kwargs = {})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 1, %convert_element_type_1, %index_put), kwargs = {})
triton_poi_fused__to_copy_scatter_add_1 = async_compile.triton('triton_poi_fused__to_copy_scatter_add_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_scatter_add_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_scatter_add_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cj/ccj2vbkeqmpk4ypm35htgbvta6atf42n63adchdhjpqrwhu7jsif.py
# Topologically Sorted Source Nodes: [corresponding_map, cat_1, indices, invalid, setitem, scatter_add_], Original ATen: [aten._to_copy, aten.cat, aten.lift_fresh, aten.index_put, aten.scatter_add]
# Source node to ATen node mapping:
#   cat_1 => cat_1
#   corresponding_map => full_default
#   indices => convert_element_type_1
#   invalid => cat
#   scatter_add_ => scatter_add
#   setitem => full_default_1, index_put
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 16], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_3, %add_4, %add_5, %add_6], 1), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.int64), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%bitwise_or, %bitwise_or_1, %bitwise_or_2, %bitwise_or_3], 1), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%cat_2, [%cat], %full_default_1), kwargs = {})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 1, %convert_element_type_1, %index_put), kwargs = {})
triton_poi_fused__to_copy_cat_index_put_lift_fresh_scatter_add_2 = async_compile.triton('triton_poi_fused__to_copy_cat_index_put_lift_fresh_scatter_add_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_index_put_lift_fresh_scatter_add_2', 'mutated_arg_names': ['in_ptr2', 'out_ptr3', 'out_ptr4'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_index_put_lift_fresh_scatter_add_2(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp149 = tl.load(in_ptr2 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr0 + (16*tmp8 + (((x0) % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr1 + (16*tmp8 + 32*x1 + (((x0) % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.floor(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 3.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp6 < tmp6
    tmp20 = tl.where(tmp19, tmp6, tmp5)
    tmp21 = tl.load(in_ptr0 + (16*tmp20 + (((x0) % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr1 + (16*tmp20 + 32*x1 + (((x0) % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.floor(tmp23)
    tmp25 = tmp24 + tmp13
    tmp26 = triton_helpers.maximum(tmp25, tmp15)
    tmp27 = triton_helpers.minimum(tmp26, tmp17)
    tmp28 = 4.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp18 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp4, tmp30, tmp31)
    tmp33 = tmp0 >= tmp3
    tmp34 = tl.full([1], 32, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tmp33 & tmp35
    tmp37 = tl.full([1], 0, tl.int64)
    tmp38 = tl.full([1], 1, tl.int64)
    tmp39 = tmp37 < tmp38
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tmp41 = tl.load(in_ptr0 + (16*tmp40 + ((((-16) + x0) % 16))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr1 + (16*tmp40 + 32*x1 + ((((-16) + x0) % 16))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.floor(tmp43)
    tmp45 = 1.0
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 3.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tmp38 < tmp38
    tmp52 = tl.where(tmp51, tmp38, tmp37)
    tmp53 = tl.load(in_ptr0 + (16*tmp52 + ((((-16) + x0) % 16))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr1 + (16*tmp52 + 32*x1 + ((((-16) + x0) % 16))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 + tmp54
    tmp56 = libdevice.floor(tmp55)
    tmp57 = triton_helpers.maximum(tmp56, tmp47)
    tmp58 = triton_helpers.minimum(tmp57, tmp49)
    tmp59 = 4.0
    tmp60 = tmp58 * tmp59
    tmp61 = tmp50 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp36, tmp61, tmp62)
    tmp64 = tmp0 >= tmp34
    tmp65 = tl.full([1], 48, tl.int64)
    tmp66 = tmp0 < tmp65
    tmp67 = tmp64 & tmp66
    tmp68 = tl.full([1], 0, tl.int64)
    tmp69 = tl.full([1], 1, tl.int64)
    tmp70 = tmp68 < tmp69
    tmp71 = tl.where(tmp70, tmp69, tmp68)
    tmp72 = tl.load(in_ptr0 + (16*tmp71 + ((((-32) + x0) % 16))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.load(in_ptr1 + (16*tmp71 + 32*x1 + ((((-32) + x0) % 16))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = tmp72 + tmp73
    tmp75 = libdevice.floor(tmp74)
    tmp76 = 0.0
    tmp77 = triton_helpers.maximum(tmp75, tmp76)
    tmp78 = 3.0
    tmp79 = triton_helpers.minimum(tmp77, tmp78)
    tmp80 = tmp69 < tmp69
    tmp81 = tl.where(tmp80, tmp69, tmp68)
    tmp82 = tl.load(in_ptr0 + (16*tmp81 + ((((-32) + x0) % 16))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tl.load(in_ptr1 + (16*tmp81 + 32*x1 + ((((-32) + x0) % 16))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp84 = tmp82 + tmp83
    tmp85 = libdevice.floor(tmp84)
    tmp86 = 1.0
    tmp87 = tmp85 + tmp86
    tmp88 = triton_helpers.maximum(tmp87, tmp76)
    tmp89 = triton_helpers.minimum(tmp88, tmp78)
    tmp90 = 4.0
    tmp91 = tmp89 * tmp90
    tmp92 = tmp79 + tmp91
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp67, tmp92, tmp93)
    tmp95 = tmp0 >= tmp65
    tmp96 = tl.full([1], 64, tl.int64)
    tmp97 = tmp0 < tmp96
    tmp98 = tl.full([1], 0, tl.int64)
    tmp99 = tl.full([1], 1, tl.int64)
    tmp100 = tmp98 < tmp99
    tmp101 = tl.where(tmp100, tmp99, tmp98)
    tmp102 = tl.load(in_ptr0 + (16*tmp101 + ((((-48) + x0) % 16))), tmp95 & xmask, eviction_policy='evict_last', other=0.0)
    tmp103 = tl.load(in_ptr1 + (16*tmp101 + 32*x1 + ((((-48) + x0) % 16))), tmp95 & xmask, eviction_policy='evict_last', other=0.0)
    tmp104 = tmp102 + tmp103
    tmp105 = libdevice.floor(tmp104)
    tmp106 = 0.0
    tmp107 = triton_helpers.maximum(tmp105, tmp106)
    tmp108 = 3.0
    tmp109 = triton_helpers.minimum(tmp107, tmp108)
    tmp110 = tmp99 < tmp99
    tmp111 = tl.where(tmp110, tmp99, tmp98)
    tmp112 = tl.load(in_ptr0 + (16*tmp111 + ((((-48) + x0) % 16))), tmp95 & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tl.load(in_ptr1 + (16*tmp111 + 32*x1 + ((((-48) + x0) % 16))), tmp95 & xmask, eviction_policy='evict_last', other=0.0)
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.floor(tmp114)
    tmp116 = triton_helpers.maximum(tmp115, tmp106)
    tmp117 = triton_helpers.minimum(tmp116, tmp108)
    tmp118 = 4.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp109 + tmp119
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp95, tmp120, tmp121)
    tmp123 = tl.where(tmp67, tmp94, tmp122)
    tmp124 = tl.where(tmp36, tmp63, tmp123)
    tmp125 = tl.where(tmp4, tmp32, tmp124)
    tmp126 = tmp14 != tmp18
    tmp127 = tmp25 != tmp27
    tmp128 = tmp126 | tmp127
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp4, tmp128, tmp129)
    tmp131 = tmp46 != tmp50
    tmp132 = tmp56 != tmp58
    tmp133 = tmp131 | tmp132
    tmp134 = tl.full(tmp133.shape, 0.0, tmp133.dtype)
    tmp135 = tl.where(tmp36, tmp133, tmp134)
    tmp136 = tmp75 != tmp79
    tmp137 = tmp87 != tmp89
    tmp138 = tmp136 | tmp137
    tmp139 = tl.full(tmp138.shape, 0.0, tmp138.dtype)
    tmp140 = tl.where(tmp67, tmp138, tmp139)
    tmp141 = tmp105 != tmp109
    tmp142 = tmp115 != tmp117
    tmp143 = tmp141 | tmp142
    tmp144 = tl.full(tmp143.shape, 0.0, tmp143.dtype)
    tmp145 = tl.where(tmp95, tmp143, tmp144)
    tmp146 = tl.where(tmp67, tmp140, tmp145)
    tmp147 = tl.where(tmp36, tmp135, tmp146)
    tmp148 = tl.where(tmp4, tmp130, tmp147)
    tmp150 = 0.0
    tmp151 = tl.where(tmp148, tmp150, tmp149)
    tmp152 = tmp125.to(tl.int64)
    tl.device_assert(((0 <= tmp152) & (tmp152 < 16)) | ~(xmask), "index out of bounds: 0 <= tmp152 < 16")
    tl.store(out_ptr3 + (x2), tmp151, xmask)
    tl.atomic_add(out_ptr4 + (tmp152 + 16*x1), tmp151, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: inductor_cache/6v/c6v33r7xf3gqvlrtz2nkeusl2wpdpwzysnppvyw6aupkw7xepnyq.py
# Topologically Sorted Source Nodes: [gt, occu_mask], Original ATen: [aten.gt, aten._to_copy]
# Source node to ATen node mapping:
#   gt => gt
#   occu_mask => convert_element_type_2
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%unsqueeze_1, 0.95), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt, torch.float32), kwargs = {})
triton_poi_fused__to_copy_gt_3 = async_compile.triton('triton_poi_fused__to_copy_gt_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_gt_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_gt_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.95
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 2, 4, 4), (32, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 2, 4, 4), (32, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf5 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        buf1 = reinterpret_tensor(buf5, (4, 16), (64, 1), 0)  # alias
        buf2 = reinterpret_tensor(buf5, (4, 16), (64, 1), 16)  # alias
        buf3 = reinterpret_tensor(buf5, (4, 16), (64, 1), 32)  # alias
        buf4 = reinterpret_tensor(buf5, (4, 16), (64, 1), 48)  # alias
        # Topologically Sorted Source Nodes: [x1, x0, x_ceil, y1, y0, y_ceil, y_floor, x_floor, sub, abs_1, sub_1, sub_2, abs_2, sub_3, mul_4, sub_4, abs_3, sub_5, sub_6, abs_4, sub_7, mul_5, sub_8, abs_5, sub_9, sub_10, abs_6, sub_11, mul_6, sub_12, abs_7, sub_13, sub_14, abs_8, sub_15, mul_7], Original ATen: [aten.floor, aten.add, aten.clamp, aten.sub, aten.abs, aten.rsub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_clamp_floor_mul_rsub_sub_0.run(arg0_1, arg1_1, buf1, buf2, buf3, buf4, 64, grid=grid(64), stream=stream0)
        buf9 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [corresponding_map, indices, scatter_add_], Original ATen: [aten._to_copy, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_scatter_add_1.run(buf9, 64, grid=grid(64), stream=stream0)
        del buf1
        del buf2
        del buf3
        del buf4
        # Topologically Sorted Source Nodes: [corresponding_map, cat_1, indices, invalid, setitem, scatter_add_], Original ATen: [aten._to_copy, aten.cat, aten.lift_fresh, aten.index_put, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_index_put_lift_fresh_scatter_add_2.run(arg0_1, arg1_1, buf5, buf5, buf9, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del arg1_1
        del buf5
        buf11 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [gt, occu_mask], Original ATen: [aten.gt, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_gt_3.run(buf9, buf11, 64, grid=grid(64), stream=stream0)
    return (buf11, reinterpret_tensor(buf9, (4, 1, 4, 4), (16, 16, 4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
