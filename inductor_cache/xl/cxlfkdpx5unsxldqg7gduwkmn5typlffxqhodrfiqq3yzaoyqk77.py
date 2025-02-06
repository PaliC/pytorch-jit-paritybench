# AOT ID: ['54_inference']
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


# kernel path: inductor_cache/zp/czpn2gzkowgapo4tv7fx2fkirklqn26v63v4dgax6yr4eyh7a4bx.py
# Topologically Sorted Source Nodes: [grid], Original ATen: [aten.affine_grid_generator]
# Source node to ATen node mapping:
#   grid => mul_4, sum_1
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %unsqueeze), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [-2]), kwargs = {})
triton_poi_fused_affine_grid_generator_0 = async_compile.triton('triton_poi_fused_affine_grid_generator_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_affine_grid_generator_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_affine_grid_generator_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 2) % 50176)
    x0 = (xindex % 2)
    x2 = xindex // 100352
    tmp46 = tl.load(in_ptr0 + (3*x0 + 6*x2), None, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr0 + (1 + 3*x0 + 6*x2), None, eviction_policy='evict_last')
    tmp132 = tl.load(in_ptr0 + (2 + 3*x0 + 6*x2), None, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = ((((x3 // 2) % 50176)) % 224)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 112.0
    tmp6 = tmp4 < tmp5
    tmp7 = 0.008928571428571428
    tmp8 = tmp4 * tmp7
    tmp9 = -0.9955357142857143
    tmp10 = tmp8 + tmp9
    tmp11 = 223 + ((-1)*((x1 % 224)))
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp7
    tmp14 = 0.9955357142857143
    tmp15 = tmp14 - tmp13
    tmp16 = tl.where(tmp6, tmp10, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = tl.full([1], -1, tl.int64)
    tmp20 = tmp19 >= tmp0
    tmp21 = tmp19 < tmp1
    tmp22 = tmp20 & tmp21
    tmp23 = x1 // 224
    tmp24 = tmp23.to(tl.float32)
    tmp25 = 112.0
    tmp26 = tmp24 < tmp25
    tmp27 = 0.008928571428571428
    tmp28 = tmp24 * tmp27
    tmp29 = -0.9955357142857143
    tmp30 = tmp28 + tmp29
    tmp31 = 223 + ((-1)*(x1 // 224))
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp27
    tmp34 = 0.9955357142857143
    tmp35 = tmp34 - tmp33
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp22, tmp36, tmp37)
    tmp39 = tmp18 + tmp38
    tmp40 = tl.full([1], -2, tl.int64)
    tmp41 = tmp40 >= tmp0
    tmp42 = 1.0
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp39 + tmp44
    tmp47 = tmp45 * tmp46
    tmp48 = tmp1 < tmp1
    tmp49 = ((((x3 // 2) % 50176)) % 224)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = 112.0
    tmp52 = tmp50 < tmp51
    tmp53 = 0.008928571428571428
    tmp54 = tmp50 * tmp53
    tmp55 = -0.9955357142857143
    tmp56 = tmp54 + tmp55
    tmp57 = 223 + ((-1)*((x1 % 224)))
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58 * tmp53
    tmp60 = 0.9955357142857143
    tmp61 = tmp60 - tmp59
    tmp62 = tl.where(tmp52, tmp56, tmp61)
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp48, tmp62, tmp63)
    tmp65 = tmp0 >= tmp0
    tmp66 = tmp65 & tmp2
    tmp67 = x1 // 224
    tmp68 = tmp67.to(tl.float32)
    tmp69 = 112.0
    tmp70 = tmp68 < tmp69
    tmp71 = 0.008928571428571428
    tmp72 = tmp68 * tmp71
    tmp73 = -0.9955357142857143
    tmp74 = tmp72 + tmp73
    tmp75 = 223 + ((-1)*(x1 // 224))
    tmp76 = tmp75.to(tl.float32)
    tmp77 = tmp76 * tmp71
    tmp78 = 0.9955357142857143
    tmp79 = tmp78 - tmp77
    tmp80 = tl.where(tmp70, tmp74, tmp79)
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp66, tmp80, tmp81)
    tmp83 = tmp64 + tmp82
    tmp84 = 1.0
    tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
    tmp86 = tl.where(tmp20, tmp84, tmp85)
    tmp87 = tmp83 + tmp86
    tmp89 = tmp87 * tmp88
    tmp90 = tmp47 + tmp89
    tmp91 = tl.full([1], 2, tl.int64)
    tmp92 = tmp91 < tmp1
    tmp93 = ((((x3 // 2) % 50176)) % 224)
    tmp94 = tmp93.to(tl.float32)
    tmp95 = 112.0
    tmp96 = tmp94 < tmp95
    tmp97 = 0.008928571428571428
    tmp98 = tmp94 * tmp97
    tmp99 = -0.9955357142857143
    tmp100 = tmp98 + tmp99
    tmp101 = 223 + ((-1)*((x1 % 224)))
    tmp102 = tmp101.to(tl.float32)
    tmp103 = tmp102 * tmp97
    tmp104 = 0.9955357142857143
    tmp105 = tmp104 - tmp103
    tmp106 = tl.where(tmp96, tmp100, tmp105)
    tmp107 = tl.full(tmp106.shape, 0.0, tmp106.dtype)
    tmp108 = tl.where(tmp92, tmp106, tmp107)
    tmp109 = tmp1 >= tmp0
    tmp110 = tmp109 & tmp48
    tmp111 = x1 // 224
    tmp112 = tmp111.to(tl.float32)
    tmp113 = 112.0
    tmp114 = tmp112 < tmp113
    tmp115 = 0.008928571428571428
    tmp116 = tmp112 * tmp115
    tmp117 = -0.9955357142857143
    tmp118 = tmp116 + tmp117
    tmp119 = 223 + ((-1)*(x1 // 224))
    tmp120 = tmp119.to(tl.float32)
    tmp121 = tmp120 * tmp115
    tmp122 = 0.9955357142857143
    tmp123 = tmp122 - tmp121
    tmp124 = tl.where(tmp114, tmp118, tmp123)
    tmp125 = tl.full(tmp124.shape, 0.0, tmp124.dtype)
    tmp126 = tl.where(tmp110, tmp124, tmp125)
    tmp127 = tmp108 + tmp126
    tmp128 = 1.0
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp65, tmp128, tmp129)
    tmp131 = tmp127 + tmp130
    tmp133 = tmp131 * tmp132
    tmp134 = tmp90 + tmp133
    tl.store(out_ptr0 + (x3), tmp134, None)
''', device_str='cuda')


# kernel path: inductor_cache/bg/cbgpj6gdmdxonyndx2gg6smoifzo4c24xla4psoi25r77pa7wwdc.py
# Topologically Sorted Source Nodes: [warped_image_batch], Original ATen: [aten.grid_sampler_2d]
# Source node to ATen node mapping:
#   warped_image_batch => add_10, add_4, add_5, add_6, add_7, add_8, add_9, floor, floor_1, full_default_12, full_default_3, full_default_6, full_default_9, ge, ge_1, ge_2, ge_3, ge_4, ge_5, ge_6, ge_7, index, index_1, index_2, index_3, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_3, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt_2, lt_3, lt_4, lt_5, lt_6, lt_7, lt_8, lt_9, mul_10, mul_11, mul_12, mul_13, mul_14, mul_5, mul_6, mul_7, mul_8, mul_9, sub_10, sub_11, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9, where_10, where_13, where_4, where_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, 2.0), kwargs = {})
#   %add_4 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, 1.5), kwargs = {})
#   %floor : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_4,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, 2.0), kwargs = {})
#   %add_5 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, 1.5), kwargs = {})
#   %floor_1 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_5,), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_3 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_1, %lt_3), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_2, %logical_and), kwargs = {})
#   %logical_and_2 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %logical_and_1), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%view_5, %view_6, %where_3, %where_2]), kwargs = {})
#   %add_6 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor, 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %add_4), kwargs = {})
#   %add_7 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_1, 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %add_5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %sub_5), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %mul_7, %full_default_3), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %where_4), kwargs = {})
#   %ge_2 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_6, 0), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_6, 4), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_5 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_3, %lt_5), kwargs = {})
#   %logical_and_4 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_4, %logical_and_3), kwargs = {})
#   %logical_and_5 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_2, %logical_and_4), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%view_5, %view_6, %where_6, %where_5]), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %floor), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %add_5), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %sub_7), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_5, %mul_8, %full_default_6), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_1, %where_7), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %mul_12), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt_6 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_7, 0), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_7, 4), kwargs = {})
#   %logical_and_6 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_5, %lt_7), kwargs = {})
#   %logical_and_7 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_6, %logical_and_6), kwargs = {})
#   %logical_and_8 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_4, %logical_and_7), kwargs = {})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%view_5, %view_6, %where_9, %where_8]), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %add_4), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %floor_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %sub_9), kwargs = {})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_8, %mul_9, %full_default_9), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_2, %where_10), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %mul_13), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_6, 0), kwargs = {})
#   %lt_8 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_6, 4), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_7, 0), kwargs = {})
#   %lt_9 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_7, 4), kwargs = {})
#   %logical_and_9 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_7, %lt_9), kwargs = {})
#   %logical_and_10 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_8, %logical_and_9), kwargs = {})
#   %logical_and_11 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_6, %logical_and_10), kwargs = {})
#   %index_3 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%view_5, %view_6, %where_12, %where_11]), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %floor), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %floor_1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %sub_11), kwargs = {})
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_13 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_11, %mul_10, %full_default_12), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_3, %where_13), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %mul_14), kwargs = {})
triton_poi_fused_grid_sampler_2d_1 = async_compile.triton('triton_poi_fused_grid_sampler_2d_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 50176)
    x2 = xindex // 200704
    x4 = xindex // 50176
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 100352*x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 2*x0 + 100352*x2), None, eviction_policy='evict_last')
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1.5
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.floor(tmp4)
    tmp6 = 0.0
    tmp7 = tmp5 >= tmp6
    tmp8 = 4.0
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
    tmp22 = tl.full([XBLOCK], 4, tl.int32)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp21 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp21)
    tl.device_assert((0 <= tmp25) & (tmp25 < 4), "index out of bounds: 0 <= tmp25 < 4")
    tmp27 = tmp5.to(tl.int64)
    tmp28 = tl.where(tmp18, tmp27, tmp20)
    tmp29 = tmp28 + tmp22
    tmp30 = tmp28 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp28)
    tl.device_assert((0 <= tmp31) & (tmp31 < 4), "index out of bounds: 0 <= tmp31 < 4")
    tmp33 = tl.load(in_ptr1 + (tmp31 + 4*tmp25 + 16*x4), None, eviction_policy='evict_last')
    tmp34 = 1.0
    tmp35 = tmp5 + tmp34
    tmp36 = tmp35 - tmp4
    tmp37 = tmp13 + tmp34
    tmp38 = tmp37 - tmp12
    tmp39 = tmp36 * tmp38
    tmp40 = tl.where(tmp18, tmp39, tmp6)
    tmp41 = tmp33 * tmp40
    tmp42 = tmp35 >= tmp6
    tmp43 = tmp35 < tmp8
    tmp44 = tmp43 & tmp16
    tmp45 = tmp42 & tmp44
    tmp46 = tl.where(tmp45, tmp19, tmp20)
    tmp47 = tmp46 + tmp22
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert((0 <= tmp49) & (tmp49 < 4), "index out of bounds: 0 <= tmp49 < 4")
    tmp51 = tmp35.to(tl.int64)
    tmp52 = tl.where(tmp45, tmp51, tmp20)
    tmp53 = tmp52 + tmp22
    tmp54 = tmp52 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp52)
    tl.device_assert((0 <= tmp55) & (tmp55 < 4), "index out of bounds: 0 <= tmp55 < 4")
    tmp57 = tl.load(in_ptr1 + (tmp55 + 4*tmp49 + 16*x4), None, eviction_policy='evict_last')
    tmp58 = tmp4 - tmp5
    tmp59 = tmp58 * tmp38
    tmp60 = tl.where(tmp45, tmp59, tmp6)
    tmp61 = tmp57 * tmp60
    tmp62 = tmp37 >= tmp6
    tmp63 = tmp37 < tmp8
    tmp64 = tmp62 & tmp63
    tmp65 = tmp9 & tmp64
    tmp66 = tmp7 & tmp65
    tmp67 = tmp37.to(tl.int64)
    tmp68 = tl.where(tmp66, tmp67, tmp20)
    tmp69 = tmp68 + tmp22
    tmp70 = tmp68 < 0
    tmp71 = tl.where(tmp70, tmp69, tmp68)
    tl.device_assert((0 <= tmp71) & (tmp71 < 4), "index out of bounds: 0 <= tmp71 < 4")
    tmp73 = tl.where(tmp66, tmp27, tmp20)
    tmp74 = tmp73 + tmp22
    tmp75 = tmp73 < 0
    tmp76 = tl.where(tmp75, tmp74, tmp73)
    tl.device_assert((0 <= tmp76) & (tmp76 < 4), "index out of bounds: 0 <= tmp76 < 4")
    tmp78 = tl.load(in_ptr1 + (tmp76 + 4*tmp71 + 16*x4), None, eviction_policy='evict_last')
    tmp79 = tmp12 - tmp13
    tmp80 = tmp36 * tmp79
    tmp81 = tl.where(tmp66, tmp80, tmp6)
    tmp82 = tmp78 * tmp81
    tmp83 = tmp43 & tmp64
    tmp84 = tmp42 & tmp83
    tmp85 = tl.where(tmp84, tmp67, tmp20)
    tmp86 = tmp85 + tmp22
    tmp87 = tmp85 < 0
    tmp88 = tl.where(tmp87, tmp86, tmp85)
    tl.device_assert((0 <= tmp88) & (tmp88 < 4), "index out of bounds: 0 <= tmp88 < 4")
    tmp90 = tl.where(tmp84, tmp51, tmp20)
    tmp91 = tmp90 + tmp22
    tmp92 = tmp90 < 0
    tmp93 = tl.where(tmp92, tmp91, tmp90)
    tl.device_assert((0 <= tmp93) & (tmp93 < 4), "index out of bounds: 0 <= tmp93 < 4")
    tmp95 = tl.load(in_ptr1 + (tmp93 + 4*tmp88 + 16*x4), None, eviction_policy='evict_last')
    tmp96 = tmp58 * tmp79
    tmp97 = tl.where(tmp84, tmp96, tmp6)
    tmp98 = tmp95 * tmp97
    tmp99 = tmp41 + tmp61
    tmp100 = tmp99 + tmp82
    tmp101 = tmp100 + tmp98
    tl.store(in_out_ptr0 + (x3), tmp101, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 2, 3), (6, 3, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((4, 50176, 2), (100352, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [grid], Original ATen: [aten.affine_grid_generator]
        stream0 = get_raw_stream(0)
        triton_poi_fused_affine_grid_generator_0.run(arg0_1, buf1, 401408, grid=grid(401408), stream=stream0)
        del arg0_1
        buf2 = empty_strided_cuda((4, 4, 224, 224), (200704, 50176, 224, 1), torch.float32)
        buf6 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [warped_image_batch], Original ATen: [aten.grid_sampler_2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_1.run(buf6, buf1, arg1_1, 802816, grid=grid(802816), stream=stream0)
        del arg1_1
        del buf1
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 2, 3), (6, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
