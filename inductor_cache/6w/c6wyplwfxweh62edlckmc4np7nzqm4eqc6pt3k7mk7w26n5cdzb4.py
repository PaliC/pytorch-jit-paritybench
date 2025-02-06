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


# kernel path: inductor_cache/nq/cnqboyu6qhunakqt6sioyyefjcz3txnoaukraaxglz5mqrenf5ks.py
# Topologically Sorted Source Nodes: [grid], Original ATen: [aten.affine_grid_generator]
# Source node to ATen node mapping:
#   grid => mul_6
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %unsqueeze), kwargs = {})
triton_poi_fused_affine_grid_generator_0 = async_compile.triton('triton_poi_fused_affine_grid_generator_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_affine_grid_generator_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_affine_grid_generator_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x5 = xindex
    x2 = ((xindex // 6) % 16)
    x1 = ((xindex // 3) % 2)
    x3 = xindex // 96
    x4 = (xindex % 6)
    tmp51 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = ((((x5 // 6) % 16)) % 4)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 2.0
    tmp6 = tmp4 < tmp5
    tmp7 = 0.5
    tmp8 = tmp4 * tmp7
    tmp9 = -0.75
    tmp10 = tmp8 + tmp9
    tmp11 = 3 + ((-1)*((x2 % 4)))
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp7
    tmp14 = 0.75
    tmp15 = tmp14 - tmp13
    tmp16 = tl.where(tmp6, tmp10, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = (-1) + x0
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tmp19 >= tmp20
    tmp22 = tmp19 < tmp1
    tmp23 = tmp21 & tmp22
    tmp24 = x2 // 4
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 2.0
    tmp27 = tmp25 < tmp26
    tmp28 = 0.5
    tmp29 = tmp25 * tmp28
    tmp30 = -0.75
    tmp31 = tmp29 + tmp30
    tmp32 = 3 + ((-1)*(x2 // 4))
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp28
    tmp35 = 0.75
    tmp36 = tmp35 - tmp34
    tmp37 = tl.where(tmp27, tmp31, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp23, tmp37, tmp38)
    tmp40 = tmp18 + tmp39
    tmp41 = (-2) + x0
    tmp42 = tmp41 >= tmp20
    tmp43 = 1.0
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = tmp40 + tmp45
    tmp47 = x1
    tmp48 = tl.full([1], 0, tl.int32)
    tmp49 = tmp47 == tmp48
    tmp50 = tmp0 == tmp48
    tmp52 = 0.5
    tmp53 = tmp51 < tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = 2.0
    tmp56 = tmp54 * tmp55
    tmp57 = 1.0
    tmp58 = tmp56 - tmp57
    tmp60 = tl.where(tmp50, tmp58, tmp59)
    tmp62 = tl.where(tmp49, tmp60, tmp61)
    tmp63 = tmp46 * tmp62
    tl.store(out_ptr0 + (x5), tmp63, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bi/cbie7qjhsj6wzyfcje2ow2pljti7bbprrjbooyogrzldjgoyx3ft.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.grid_sampler_2d]
# Source node to ATen node mapping:
#   output => abs_1, abs_2, add_10, add_11, add_12, add_4, add_5, add_6, add_7, add_8, add_9, bitwise_and, bitwise_and_1, clamp_max, clamp_max_1, clamp_min, clamp_min_1, convert_element_type_5, convert_element_type_6, div, div_1, eq, eq_1, floor, floor_1, floor_2, floor_3, fmod, fmod_1, full_default_10, full_default_13, full_default_4, full_default_7, ge, ge_1, ge_2, ge_3, ge_4, ge_5, ge_6, ge_7, index, index_1, index_2, index_3, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_3, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt_10, lt_3, lt_4, lt_5, lt_6, lt_7, lt_8, lt_9, mul_10, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_7, mul_8, mul_9, sub_10, sub_11, sub_12, sub_13, sub_14, sub_15, sub_16, sub_5, sub_6, sub_7, sub_8, sub_9, where_12, where_15, where_2, where_3, where_6, where_9
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, 2.0), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, 1.5), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, -0.5), kwargs = {})
#   %abs_1 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%sub_5,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_1, 4.0), kwargs = {})
#   %floor : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%div,), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int8), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Scalar](args = (%convert_element_type_5, 1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%bitwise_and, 0), kwargs = {})
#   %fmod : [num_users=2] = call_function[target=torch.ops.aten.fmod.Scalar](args = (%abs_1, 4.0), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%fmod, -0.5), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (3.5, %fmod), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %add_5, %sub_6), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%where_2, 0), kwargs = {})
#   %clamp_max : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 3), kwargs = {})
#   %floor_2 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%clamp_max,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_2, 0), kwargs = {})
#   %lt_3 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_2, 4), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, 2.0), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, 1.5), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, -0.5), kwargs = {})
#   %abs_2 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%sub_7,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_2, 4.0), kwargs = {})
#   %floor_1 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%div_1,), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_1, torch.int8), kwargs = {})
#   %bitwise_and_1 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Scalar](args = (%convert_element_type_6, 1), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%bitwise_and_1, 0), kwargs = {})
#   %fmod_1 : [num_users=2] = call_function[target=torch.ops.aten.fmod.Scalar](args = (%abs_2, 4.0), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%fmod_1, -0.5), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (3.5, %fmod_1), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %add_7, %sub_8), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%where_3, 0), kwargs = {})
#   %clamp_max_1 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 3), kwargs = {})
#   %floor_3 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%clamp_max_1,), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_3, 0), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_3, 4), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_1, %lt_4), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_3, %logical_and), kwargs = {})
#   %logical_and_2 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %logical_and_1), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%view_5, %view_6, %where_5, %where_4]), kwargs = {})
#   %add_8 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_2, 1), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %clamp_max), kwargs = {})
#   %add_9 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_3, 1), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %clamp_max_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %sub_10), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %mul_9, %full_default_4), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %where_6), kwargs = {})
#   %ge_2 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_8, 0), kwargs = {})
#   %lt_5 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_8, 4), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_3, 0), kwargs = {})
#   %lt_6 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_3, 4), kwargs = {})
#   %logical_and_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_3, %lt_6), kwargs = {})
#   %logical_and_4 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_5, %logical_and_3), kwargs = {})
#   %logical_and_5 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_2, %logical_and_4), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%view_5, %view_6, %where_8, %where_7]), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, %floor_2), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %clamp_max_1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %sub_12), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_5, %mul_10, %full_default_7), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_1, %where_9), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_2, 0), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_2, 4), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_9, 0), kwargs = {})
#   %lt_8 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_9, 4), kwargs = {})
#   %logical_and_6 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_5, %lt_8), kwargs = {})
#   %logical_and_7 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_7, %logical_and_6), kwargs = {})
#   %logical_and_8 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_4, %logical_and_7), kwargs = {})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%view_5, %view_6, %where_11, %where_10]), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %clamp_max), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max_1, %floor_3), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %sub_14), kwargs = {})
#   %full_default_10 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_8, %mul_11, %full_default_10), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_2, %where_12), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_15), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_8, 0), kwargs = {})
#   %lt_9 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_8, 4), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_9, 0), kwargs = {})
#   %lt_10 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_9, 4), kwargs = {})
#   %logical_and_9 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_7, %lt_10), kwargs = {})
#   %logical_and_10 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_9, %logical_and_9), kwargs = {})
#   %logical_and_11 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_6, %logical_and_10), kwargs = {})
#   %index_3 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%view_5, %view_6, %where_14, %where_13]), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, %floor_2), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max_1, %floor_3), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %sub_16), kwargs = {})
#   %full_default_13 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_15 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_11, %mul_12, %full_default_13), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_3, %where_15), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mul_16), kwargs = {})
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp0 = tl.load(in_ptr0 + (3 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (4 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (5 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (1 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (2 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tmp7 = 1.5
    tmp8 = tmp6 + tmp7
    tmp9 = -0.5
    tmp10 = tmp8 - tmp9
    tmp11 = tl_math.abs(tmp10)
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = libdevice.floor(tmp13)
    tmp15 = tmp14.to(tl.int8)
    tmp16 = tl.full([1], 1, tl.int8)
    tmp17 = tmp15 & tmp16
    tmp18 = tl.full([1], 0, tl.int8)
    tmp19 = tmp17 == tmp18
    tmp20 = 4.0
    tmp21 = libdevice.fmod(tmp11, tmp20)
    tmp22 = tmp21 + tmp9
    tmp23 = 3.5
    tmp24 = tmp23 - tmp21
    tmp25 = tl.where(tmp19, tmp22, tmp24)
    tmp26 = 0.0
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp28 = 3.0
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = libdevice.floor(tmp29)
    tmp33 = tmp31 + tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp35 * tmp5
    tmp37 = tmp36 + tmp7
    tmp38 = tmp37 - tmp9
    tmp39 = tl_math.abs(tmp38)
    tmp40 = tmp39 * tmp12
    tmp41 = libdevice.floor(tmp40)
    tmp42 = tmp41.to(tl.int8)
    tmp43 = tmp42 & tmp16
    tmp44 = tmp43 == tmp18
    tmp45 = libdevice.fmod(tmp39, tmp20)
    tmp46 = tmp45 + tmp9
    tmp47 = tmp23 - tmp45
    tmp48 = tl.where(tmp44, tmp46, tmp47)
    tmp49 = triton_helpers.maximum(tmp48, tmp26)
    tmp50 = triton_helpers.minimum(tmp49, tmp28)
    tmp51 = libdevice.floor(tmp50)
    tmp52 = 1.0
    tmp53 = tmp51 + tmp52
    tmp54 = tmp53 - tmp50
    tmp55 = tmp29 - tmp30
    tmp56 = tmp30 + tmp52
    tmp57 = tmp56 - tmp29
    tmp58 = tmp50 - tmp51
    tmp59 = tmp51 >= tmp26
    tmp60 = tmp51 < tmp20
    tmp61 = tmp30 >= tmp26
    tmp62 = tmp30 < tmp20
    tmp63 = tmp61 & tmp62
    tmp64 = tmp60 & tmp63
    tmp65 = tmp59 & tmp64
    tmp66 = tmp30.to(tl.int64)
    tmp67 = tl.full([1], 0, tl.int64)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tl.full([XBLOCK], 4, tl.int32)
    tmp70 = tmp68 + tmp69
    tmp71 = tmp68 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp68)
    tl.device_assert(((0 <= tmp72) & (tmp72 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp72 < 4")
    tmp74 = tmp51.to(tl.int64)
    tmp75 = tl.where(tmp65, tmp74, tmp67)
    tmp76 = tmp75 + tmp69
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tl.device_assert(((0 <= tmp78) & (tmp78 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp78 < 4")
    tmp80 = tl.load(in_ptr1 + (tmp78 + 4*tmp72 + 16*x4), xmask, eviction_policy='evict_last')
    tmp81 = tmp54 * tmp57
    tmp82 = tl.where(tmp65, tmp81, tmp26)
    tmp83 = tmp80 * tmp82
    tmp84 = tmp53 >= tmp26
    tmp85 = tmp53 < tmp20
    tmp86 = tmp85 & tmp63
    tmp87 = tmp84 & tmp86
    tmp88 = tl.where(tmp87, tmp66, tmp67)
    tmp89 = tmp88 + tmp69
    tmp90 = tmp88 < 0
    tmp91 = tl.where(tmp90, tmp89, tmp88)
    tl.device_assert(((0 <= tmp91) & (tmp91 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp91 < 4")
    tmp93 = tmp53.to(tl.int64)
    tmp94 = tl.where(tmp87, tmp93, tmp67)
    tmp95 = tmp94 + tmp69
    tmp96 = tmp94 < 0
    tmp97 = tl.where(tmp96, tmp95, tmp94)
    tl.device_assert(((0 <= tmp97) & (tmp97 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp97 < 4")
    tmp99 = tl.load(in_ptr1 + (tmp97 + 4*tmp91 + 16*x4), xmask, eviction_policy='evict_last')
    tmp100 = tmp58 * tmp57
    tmp101 = tl.where(tmp87, tmp100, tmp26)
    tmp102 = tmp99 * tmp101
    tmp103 = tmp83 + tmp102
    tmp104 = tmp56 >= tmp26
    tmp105 = tmp56 < tmp20
    tmp106 = tmp104 & tmp105
    tmp107 = tmp60 & tmp106
    tmp108 = tmp59 & tmp107
    tmp109 = tmp56.to(tl.int64)
    tmp110 = tl.where(tmp108, tmp109, tmp67)
    tmp111 = tmp110 + tmp69
    tmp112 = tmp110 < 0
    tmp113 = tl.where(tmp112, tmp111, tmp110)
    tl.device_assert(((0 <= tmp113) & (tmp113 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp113 < 4")
    tmp115 = tl.where(tmp108, tmp74, tmp67)
    tmp116 = tmp115 + tmp69
    tmp117 = tmp115 < 0
    tmp118 = tl.where(tmp117, tmp116, tmp115)
    tl.device_assert(((0 <= tmp118) & (tmp118 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp118 < 4")
    tmp120 = tl.load(in_ptr1 + (tmp118 + 4*tmp113 + 16*x4), xmask, eviction_policy='evict_last')
    tmp121 = tmp54 * tmp55
    tmp122 = tl.where(tmp108, tmp121, tmp26)
    tmp123 = tmp120 * tmp122
    tmp124 = tmp103 + tmp123
    tmp125 = tmp85 & tmp106
    tmp126 = tmp84 & tmp125
    tmp127 = tl.where(tmp126, tmp109, tmp67)
    tmp128 = tmp127 + tmp69
    tmp129 = tmp127 < 0
    tmp130 = tl.where(tmp129, tmp128, tmp127)
    tl.device_assert(((0 <= tmp130) & (tmp130 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp130 < 4")
    tmp132 = tl.where(tmp126, tmp93, tmp67)
    tmp133 = tmp132 + tmp69
    tmp134 = tmp132 < 0
    tmp135 = tl.where(tmp134, tmp133, tmp132)
    tl.device_assert(((0 <= tmp135) & (tmp135 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp135 < 4")
    tmp137 = tl.load(in_ptr1 + (tmp135 + 4*tmp130 + 16*x4), xmask, eviction_policy='evict_last')
    tmp138 = tmp58 * tmp55
    tmp139 = tl.where(tmp126, tmp138, tmp26)
    tmp140 = tmp137 * tmp139
    tmp141 = tmp124 + tmp140
    tl.store(in_out_ptr0 + (x3), tmp141, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (2, 3), (3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [bernoulli], Original ATen: [aten.bernoulli]
        buf0 = torch.ops.aten.rand.default([4], dtype=torch.float32, device=device(type='cuda', index=0), pin_memory=False)
        buf1 = buf0
        del buf0
        buf3 = empty_strided_cuda((4, 16, 3, 2), (96, 6, 1, 3), torch.float32)
        # Topologically Sorted Source Nodes: [grid], Original ATen: [aten.affine_grid_generator]
        stream0 = get_raw_stream(0)
        triton_poi_fused_affine_grid_generator_0.run(buf1, arg1_1, buf3, 384, grid=grid(384), stream=stream0)
        del arg1_1
        del buf1
        buf6 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf10 = buf6; del buf6  # reuse
        buf15 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.grid_sampler_2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_1.run(buf15, buf3, arg0_1, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del buf3
    return (buf15, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
