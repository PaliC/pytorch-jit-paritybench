# AOT ID: ['115_forward']
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


# kernel path: inductor_cache/in/cinbtjb3wzgcalqsafqwntomsd7unebj7ez5f2dbwzu23zb7yri6.py
# Topologically Sorted Source Nodes: [batch_norm, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   x => mul_3, sigmoid
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ts/ctsmpumlqkxpxzjcimvitf6vueetfs6mllye2rg5nhd4wdlxraky.py
# Topologically Sorted Source Nodes: [y1, cat], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
# Source node to ATen node mapping:
#   cat => cat
#   y1 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_3, %getitem, %getitem_2, %getitem_4], 1), kwargs = {})
triton_poi_fused_cat_max_pool2d_with_indices_1 = async_compile.triton('triton_poi_fused_cat_max_pool2d_with_indices_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x6 = xindex
    x3 = (xindex % 32)
    x4 = xindex // 32
    tmp189 = tl.load(in_ptr0 + (x6), xmask)
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-2) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-10) + x6), tmp10 & xmask, other=float("-inf"))
    tmp12 = (-1) + x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-9) + x6), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8) + x6), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 1 + x0
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tl.load(in_ptr0 + ((-7) + x6), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = 2 + x0
    tmp34 = tmp33 >= tmp1
    tmp35 = tmp33 < tmp3
    tmp36 = tmp34 & tmp35
    tmp37 = tmp5 & tmp36
    tmp38 = tl.load(in_ptr0 + ((-6) + x6), tmp37 & xmask, other=float("-inf"))
    tmp39 = triton_helpers.maximum(tmp38, tmp32)
    tmp40 = (-1) + x1
    tmp41 = tmp40 >= tmp1
    tmp42 = tmp40 < tmp3
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp9
    tmp45 = tl.load(in_ptr0 + ((-6) + x6), tmp44 & xmask, other=float("-inf"))
    tmp46 = triton_helpers.maximum(tmp45, tmp39)
    tmp47 = tmp43 & tmp15
    tmp48 = tl.load(in_ptr0 + ((-5) + x6), tmp47 & xmask, other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp46)
    tmp50 = tmp43 & tmp22
    tmp51 = tl.load(in_ptr0 + ((-4) + x6), tmp50 & xmask, other=float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp49)
    tmp53 = tmp43 & tmp29
    tmp54 = tl.load(in_ptr0 + ((-3) + x6), tmp53 & xmask, other=float("-inf"))
    tmp55 = triton_helpers.maximum(tmp54, tmp52)
    tmp56 = tmp43 & tmp36
    tmp57 = tl.load(in_ptr0 + ((-2) + x6), tmp56 & xmask, other=float("-inf"))
    tmp58 = triton_helpers.maximum(tmp57, tmp55)
    tmp59 = x1
    tmp60 = tmp59 >= tmp1
    tmp61 = tmp59 < tmp3
    tmp62 = tmp60 & tmp61
    tmp63 = tmp62 & tmp9
    tmp64 = tl.load(in_ptr0 + ((-2) + x6), tmp63 & xmask, other=float("-inf"))
    tmp65 = triton_helpers.maximum(tmp64, tmp58)
    tmp66 = tmp62 & tmp15
    tmp67 = tl.load(in_ptr0 + ((-1) + x6), tmp66 & xmask, other=float("-inf"))
    tmp68 = triton_helpers.maximum(tmp67, tmp65)
    tmp69 = tmp62 & tmp22
    tmp70 = tl.load(in_ptr0 + (x6), tmp69 & xmask, other=float("-inf"))
    tmp71 = triton_helpers.maximum(tmp70, tmp68)
    tmp72 = tmp62 & tmp29
    tmp73 = tl.load(in_ptr0 + (1 + x6), tmp72 & xmask, other=float("-inf"))
    tmp74 = triton_helpers.maximum(tmp73, tmp71)
    tmp75 = tmp62 & tmp36
    tmp76 = tl.load(in_ptr0 + (2 + x6), tmp75 & xmask, other=float("-inf"))
    tmp77 = triton_helpers.maximum(tmp76, tmp74)
    tmp78 = 1 + x1
    tmp79 = tmp78 >= tmp1
    tmp80 = tmp78 < tmp3
    tmp81 = tmp79 & tmp80
    tmp82 = tmp81 & tmp9
    tmp83 = tl.load(in_ptr0 + (2 + x6), tmp82 & xmask, other=float("-inf"))
    tmp84 = triton_helpers.maximum(tmp83, tmp77)
    tmp85 = tmp81 & tmp15
    tmp86 = tl.load(in_ptr0 + (3 + x6), tmp85 & xmask, other=float("-inf"))
    tmp87 = triton_helpers.maximum(tmp86, tmp84)
    tmp88 = tmp81 & tmp22
    tmp89 = tl.load(in_ptr0 + (4 + x6), tmp88 & xmask, other=float("-inf"))
    tmp90 = triton_helpers.maximum(tmp89, tmp87)
    tmp91 = tmp81 & tmp29
    tmp92 = tl.load(in_ptr0 + (5 + x6), tmp91 & xmask, other=float("-inf"))
    tmp93 = triton_helpers.maximum(tmp92, tmp90)
    tmp94 = tmp81 & tmp36
    tmp95 = tl.load(in_ptr0 + (6 + x6), tmp94 & xmask, other=float("-inf"))
    tmp96 = triton_helpers.maximum(tmp95, tmp93)
    tmp97 = 2 + x1
    tmp98 = tmp97 >= tmp1
    tmp99 = tmp97 < tmp3
    tmp100 = tmp98 & tmp99
    tmp101 = tmp100 & tmp9
    tmp102 = tl.load(in_ptr0 + (6 + x6), tmp101 & xmask, other=float("-inf"))
    tmp103 = triton_helpers.maximum(tmp102, tmp96)
    tmp104 = tmp100 & tmp15
    tmp105 = tl.load(in_ptr0 + (7 + x6), tmp104 & xmask, other=float("-inf"))
    tmp106 = triton_helpers.maximum(tmp105, tmp103)
    tmp107 = tmp100 & tmp22
    tmp108 = tl.load(in_ptr0 + (8 + x6), tmp107 & xmask, other=float("-inf"))
    tmp109 = triton_helpers.maximum(tmp108, tmp106)
    tmp110 = tmp100 & tmp29
    tmp111 = tl.load(in_ptr0 + (9 + x6), tmp110 & xmask, other=float("-inf"))
    tmp112 = triton_helpers.maximum(tmp111, tmp109)
    tmp113 = tmp100 & tmp36
    tmp114 = tl.load(in_ptr0 + (10 + x6), tmp113 & xmask, other=float("-inf"))
    tmp115 = triton_helpers.maximum(tmp114, tmp112)
    tmp116 = tmp17 > tmp11
    tmp117 = tl.full([1], 1, tl.int8)
    tmp118 = tl.full([1], 0, tl.int8)
    tmp119 = tl.where(tmp116, tmp117, tmp118)
    tmp120 = tmp24 > tmp18
    tmp121 = tl.full([1], 2, tl.int8)
    tmp122 = tl.where(tmp120, tmp121, tmp119)
    tmp123 = tmp31 > tmp25
    tmp124 = tl.full([1], 3, tl.int8)
    tmp125 = tl.where(tmp123, tmp124, tmp122)
    tmp126 = tmp38 > tmp32
    tmp127 = tl.full([1], 4, tl.int8)
    tmp128 = tl.where(tmp126, tmp127, tmp125)
    tmp129 = tmp45 > tmp39
    tmp130 = tl.full([1], 5, tl.int8)
    tmp131 = tl.where(tmp129, tmp130, tmp128)
    tmp132 = tmp48 > tmp46
    tmp133 = tl.full([1], 6, tl.int8)
    tmp134 = tl.where(tmp132, tmp133, tmp131)
    tmp135 = tmp51 > tmp49
    tmp136 = tl.full([1], 7, tl.int8)
    tmp137 = tl.where(tmp135, tmp136, tmp134)
    tmp138 = tmp54 > tmp52
    tmp139 = tl.full([1], 8, tl.int8)
    tmp140 = tl.where(tmp138, tmp139, tmp137)
    tmp141 = tmp57 > tmp55
    tmp142 = tl.full([1], 9, tl.int8)
    tmp143 = tl.where(tmp141, tmp142, tmp140)
    tmp144 = tmp64 > tmp58
    tmp145 = tl.full([1], 10, tl.int8)
    tmp146 = tl.where(tmp144, tmp145, tmp143)
    tmp147 = tmp67 > tmp65
    tmp148 = tl.full([1], 11, tl.int8)
    tmp149 = tl.where(tmp147, tmp148, tmp146)
    tmp150 = tmp70 > tmp68
    tmp151 = tl.full([1], 12, tl.int8)
    tmp152 = tl.where(tmp150, tmp151, tmp149)
    tmp153 = tmp73 > tmp71
    tmp154 = tl.full([1], 13, tl.int8)
    tmp155 = tl.where(tmp153, tmp154, tmp152)
    tmp156 = tmp76 > tmp74
    tmp157 = tl.full([1], 14, tl.int8)
    tmp158 = tl.where(tmp156, tmp157, tmp155)
    tmp159 = tmp83 > tmp77
    tmp160 = tl.full([1], 15, tl.int8)
    tmp161 = tl.where(tmp159, tmp160, tmp158)
    tmp162 = tmp86 > tmp84
    tmp163 = tl.full([1], 16, tl.int8)
    tmp164 = tl.where(tmp162, tmp163, tmp161)
    tmp165 = tmp89 > tmp87
    tmp166 = tl.full([1], 17, tl.int8)
    tmp167 = tl.where(tmp165, tmp166, tmp164)
    tmp168 = tmp92 > tmp90
    tmp169 = tl.full([1], 18, tl.int8)
    tmp170 = tl.where(tmp168, tmp169, tmp167)
    tmp171 = tmp95 > tmp93
    tmp172 = tl.full([1], 19, tl.int8)
    tmp173 = tl.where(tmp171, tmp172, tmp170)
    tmp174 = tmp102 > tmp96
    tmp175 = tl.full([1], 20, tl.int8)
    tmp176 = tl.where(tmp174, tmp175, tmp173)
    tmp177 = tmp105 > tmp103
    tmp178 = tl.full([1], 21, tl.int8)
    tmp179 = tl.where(tmp177, tmp178, tmp176)
    tmp180 = tmp108 > tmp106
    tmp181 = tl.full([1], 22, tl.int8)
    tmp182 = tl.where(tmp180, tmp181, tmp179)
    tmp183 = tmp111 > tmp109
    tmp184 = tl.full([1], 23, tl.int8)
    tmp185 = tl.where(tmp183, tmp184, tmp182)
    tmp186 = tmp114 > tmp112
    tmp187 = tl.full([1], 24, tl.int8)
    tmp188 = tl.where(tmp186, tmp187, tmp185)
    tl.store(out_ptr0 + (x6), tmp115, xmask)
    tl.store(out_ptr1 + (x6), tmp188, xmask)
    tl.store(out_ptr2 + (x3 + 128*x4), tmp189, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mr/cmrn33hh6vyumutkyw7zgyh5glvis5ihvysbfwep2kfhznroktnl.py
# Topologically Sorted Source Nodes: [max_pool2d_2, cat], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
# Source node to ATen node mapping:
#   cat => cat
#   max_pool2d_2 => _low_memory_max_pool2d_with_offsets_2, getitem_5
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%getitem_2, [5, 5], [1, 1], [2, 2], [1, 1], False), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_3, %getitem, %getitem_2, %getitem_4], 1), kwargs = {})
triton_poi_fused_cat_max_pool2d_with_indices_2 = async_compile.triton('triton_poi_fused_cat_max_pool2d_with_indices_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x7 = xindex
    x3 = xindex // 32
    x5 = (xindex % 32)
    tmp189 = tl.load(in_ptr0 + (x7), xmask)
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-2) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-10) + x7), tmp10 & xmask, other=float("-inf"))
    tmp12 = (-1) + x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-9) + x7), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8) + x7), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 1 + x0
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tl.load(in_ptr0 + ((-7) + x7), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = 2 + x0
    tmp34 = tmp33 >= tmp1
    tmp35 = tmp33 < tmp3
    tmp36 = tmp34 & tmp35
    tmp37 = tmp5 & tmp36
    tmp38 = tl.load(in_ptr0 + ((-6) + x7), tmp37 & xmask, other=float("-inf"))
    tmp39 = triton_helpers.maximum(tmp38, tmp32)
    tmp40 = (-1) + x1
    tmp41 = tmp40 >= tmp1
    tmp42 = tmp40 < tmp3
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp9
    tmp45 = tl.load(in_ptr0 + ((-6) + x7), tmp44 & xmask, other=float("-inf"))
    tmp46 = triton_helpers.maximum(tmp45, tmp39)
    tmp47 = tmp43 & tmp15
    tmp48 = tl.load(in_ptr0 + ((-5) + x7), tmp47 & xmask, other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp46)
    tmp50 = tmp43 & tmp22
    tmp51 = tl.load(in_ptr0 + ((-4) + x7), tmp50 & xmask, other=float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp49)
    tmp53 = tmp43 & tmp29
    tmp54 = tl.load(in_ptr0 + ((-3) + x7), tmp53 & xmask, other=float("-inf"))
    tmp55 = triton_helpers.maximum(tmp54, tmp52)
    tmp56 = tmp43 & tmp36
    tmp57 = tl.load(in_ptr0 + ((-2) + x7), tmp56 & xmask, other=float("-inf"))
    tmp58 = triton_helpers.maximum(tmp57, tmp55)
    tmp59 = x1
    tmp60 = tmp59 >= tmp1
    tmp61 = tmp59 < tmp3
    tmp62 = tmp60 & tmp61
    tmp63 = tmp62 & tmp9
    tmp64 = tl.load(in_ptr0 + ((-2) + x7), tmp63 & xmask, other=float("-inf"))
    tmp65 = triton_helpers.maximum(tmp64, tmp58)
    tmp66 = tmp62 & tmp15
    tmp67 = tl.load(in_ptr0 + ((-1) + x7), tmp66 & xmask, other=float("-inf"))
    tmp68 = triton_helpers.maximum(tmp67, tmp65)
    tmp69 = tmp62 & tmp22
    tmp70 = tl.load(in_ptr0 + (x7), tmp69 & xmask, other=float("-inf"))
    tmp71 = triton_helpers.maximum(tmp70, tmp68)
    tmp72 = tmp62 & tmp29
    tmp73 = tl.load(in_ptr0 + (1 + x7), tmp72 & xmask, other=float("-inf"))
    tmp74 = triton_helpers.maximum(tmp73, tmp71)
    tmp75 = tmp62 & tmp36
    tmp76 = tl.load(in_ptr0 + (2 + x7), tmp75 & xmask, other=float("-inf"))
    tmp77 = triton_helpers.maximum(tmp76, tmp74)
    tmp78 = 1 + x1
    tmp79 = tmp78 >= tmp1
    tmp80 = tmp78 < tmp3
    tmp81 = tmp79 & tmp80
    tmp82 = tmp81 & tmp9
    tmp83 = tl.load(in_ptr0 + (2 + x7), tmp82 & xmask, other=float("-inf"))
    tmp84 = triton_helpers.maximum(tmp83, tmp77)
    tmp85 = tmp81 & tmp15
    tmp86 = tl.load(in_ptr0 + (3 + x7), tmp85 & xmask, other=float("-inf"))
    tmp87 = triton_helpers.maximum(tmp86, tmp84)
    tmp88 = tmp81 & tmp22
    tmp89 = tl.load(in_ptr0 + (4 + x7), tmp88 & xmask, other=float("-inf"))
    tmp90 = triton_helpers.maximum(tmp89, tmp87)
    tmp91 = tmp81 & tmp29
    tmp92 = tl.load(in_ptr0 + (5 + x7), tmp91 & xmask, other=float("-inf"))
    tmp93 = triton_helpers.maximum(tmp92, tmp90)
    tmp94 = tmp81 & tmp36
    tmp95 = tl.load(in_ptr0 + (6 + x7), tmp94 & xmask, other=float("-inf"))
    tmp96 = triton_helpers.maximum(tmp95, tmp93)
    tmp97 = 2 + x1
    tmp98 = tmp97 >= tmp1
    tmp99 = tmp97 < tmp3
    tmp100 = tmp98 & tmp99
    tmp101 = tmp100 & tmp9
    tmp102 = tl.load(in_ptr0 + (6 + x7), tmp101 & xmask, other=float("-inf"))
    tmp103 = triton_helpers.maximum(tmp102, tmp96)
    tmp104 = tmp100 & tmp15
    tmp105 = tl.load(in_ptr0 + (7 + x7), tmp104 & xmask, other=float("-inf"))
    tmp106 = triton_helpers.maximum(tmp105, tmp103)
    tmp107 = tmp100 & tmp22
    tmp108 = tl.load(in_ptr0 + (8 + x7), tmp107 & xmask, other=float("-inf"))
    tmp109 = triton_helpers.maximum(tmp108, tmp106)
    tmp110 = tmp100 & tmp29
    tmp111 = tl.load(in_ptr0 + (9 + x7), tmp110 & xmask, other=float("-inf"))
    tmp112 = triton_helpers.maximum(tmp111, tmp109)
    tmp113 = tmp100 & tmp36
    tmp114 = tl.load(in_ptr0 + (10 + x7), tmp113 & xmask, other=float("-inf"))
    tmp115 = triton_helpers.maximum(tmp114, tmp112)
    tmp116 = tmp17 > tmp11
    tmp117 = tl.full([1], 1, tl.int8)
    tmp118 = tl.full([1], 0, tl.int8)
    tmp119 = tl.where(tmp116, tmp117, tmp118)
    tmp120 = tmp24 > tmp18
    tmp121 = tl.full([1], 2, tl.int8)
    tmp122 = tl.where(tmp120, tmp121, tmp119)
    tmp123 = tmp31 > tmp25
    tmp124 = tl.full([1], 3, tl.int8)
    tmp125 = tl.where(tmp123, tmp124, tmp122)
    tmp126 = tmp38 > tmp32
    tmp127 = tl.full([1], 4, tl.int8)
    tmp128 = tl.where(tmp126, tmp127, tmp125)
    tmp129 = tmp45 > tmp39
    tmp130 = tl.full([1], 5, tl.int8)
    tmp131 = tl.where(tmp129, tmp130, tmp128)
    tmp132 = tmp48 > tmp46
    tmp133 = tl.full([1], 6, tl.int8)
    tmp134 = tl.where(tmp132, tmp133, tmp131)
    tmp135 = tmp51 > tmp49
    tmp136 = tl.full([1], 7, tl.int8)
    tmp137 = tl.where(tmp135, tmp136, tmp134)
    tmp138 = tmp54 > tmp52
    tmp139 = tl.full([1], 8, tl.int8)
    tmp140 = tl.where(tmp138, tmp139, tmp137)
    tmp141 = tmp57 > tmp55
    tmp142 = tl.full([1], 9, tl.int8)
    tmp143 = tl.where(tmp141, tmp142, tmp140)
    tmp144 = tmp64 > tmp58
    tmp145 = tl.full([1], 10, tl.int8)
    tmp146 = tl.where(tmp144, tmp145, tmp143)
    tmp147 = tmp67 > tmp65
    tmp148 = tl.full([1], 11, tl.int8)
    tmp149 = tl.where(tmp147, tmp148, tmp146)
    tmp150 = tmp70 > tmp68
    tmp151 = tl.full([1], 12, tl.int8)
    tmp152 = tl.where(tmp150, tmp151, tmp149)
    tmp153 = tmp73 > tmp71
    tmp154 = tl.full([1], 13, tl.int8)
    tmp155 = tl.where(tmp153, tmp154, tmp152)
    tmp156 = tmp76 > tmp74
    tmp157 = tl.full([1], 14, tl.int8)
    tmp158 = tl.where(tmp156, tmp157, tmp155)
    tmp159 = tmp83 > tmp77
    tmp160 = tl.full([1], 15, tl.int8)
    tmp161 = tl.where(tmp159, tmp160, tmp158)
    tmp162 = tmp86 > tmp84
    tmp163 = tl.full([1], 16, tl.int8)
    tmp164 = tl.where(tmp162, tmp163, tmp161)
    tmp165 = tmp89 > tmp87
    tmp166 = tl.full([1], 17, tl.int8)
    tmp167 = tl.where(tmp165, tmp166, tmp164)
    tmp168 = tmp92 > tmp90
    tmp169 = tl.full([1], 18, tl.int8)
    tmp170 = tl.where(tmp168, tmp169, tmp167)
    tmp171 = tmp95 > tmp93
    tmp172 = tl.full([1], 19, tl.int8)
    tmp173 = tl.where(tmp171, tmp172, tmp170)
    tmp174 = tmp102 > tmp96
    tmp175 = tl.full([1], 20, tl.int8)
    tmp176 = tl.where(tmp174, tmp175, tmp173)
    tmp177 = tmp105 > tmp103
    tmp178 = tl.full([1], 21, tl.int8)
    tmp179 = tl.where(tmp177, tmp178, tmp176)
    tmp180 = tmp108 > tmp106
    tmp181 = tl.full([1], 22, tl.int8)
    tmp182 = tl.where(tmp180, tmp181, tmp179)
    tmp183 = tmp111 > tmp109
    tmp184 = tl.full([1], 23, tl.int8)
    tmp185 = tl.where(tmp183, tmp184, tmp182)
    tmp186 = tmp114 > tmp112
    tmp187 = tl.full([1], 24, tl.int8)
    tmp188 = tl.where(tmp186, tmp187, tmp185)
    tl.store(out_ptr0 + (x5 + 128*x3), tmp115, xmask)
    tl.store(out_ptr1 + (x7), tmp188, xmask)
    tl.store(out_ptr2 + (x5 + 128*x3), tmp189, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rw/crwsutu4qjwkb6jaep3vpr7rh2ew5s4nty3kwtp6d4e7adexszo5.py
# Topologically Sorted Source Nodes: [batch_norm_1, silu_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_5, mul_6, sub_1
#   silu_1 => mul_7, sigmoid_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (2, ), (1, ))
    assert_size_stride(primals_4, (2, ), (1, ))
    assert_size_stride(primals_5, (2, ), (1, ))
    assert_size_stride(primals_6, (2, ), (1, ))
    assert_size_stride(primals_7, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 2, 4, 4), (32, 16, 4, 1))
        buf1 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [batch_norm, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_0.run(buf2, buf0, primals_3, primals_4, primals_5, primals_6, 128, grid=grid(128), stream=stream0)
        buf3 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.int8)
        buf12 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        buf9 = reinterpret_tensor(buf12, (4, 2, 4, 4), (128, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [y1, cat], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_max_pool2d_with_indices_1.run(buf2, buf3, buf4, buf9, 128, grid=grid(128), stream=stream0)
        buf5 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        buf6 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.int8)
        buf10 = reinterpret_tensor(buf12, (4, 2, 4, 4), (128, 16, 4, 1), 32)  # alias
        # Topologically Sorted Source Nodes: [y2, cat], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_max_pool2d_with_indices_1.run(buf3, buf5, buf6, buf10, 128, grid=grid(128), stream=stream0)
        buf7 = reinterpret_tensor(buf12, (4, 2, 4, 4), (128, 16, 4, 1), 96)  # alias
        buf8 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.int8)
        buf11 = reinterpret_tensor(buf12, (4, 2, 4, 4), (128, 16, 4, 1), 64)  # alias
        # Topologically Sorted Source Nodes: [max_pool2d_2, cat], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_max_pool2d_with_indices_2.run(buf5, buf7, buf8, buf11, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 4, 4, 4), (64, 16, 4, 1))
        buf14 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_1, silu_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_3.run(buf15, buf13, primals_8, primals_9, primals_10, primals_11, 256, grid=grid(256), stream=stream0)
    return (buf15, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, buf0, buf2, buf3, buf4, buf5, buf6, buf8, buf12, buf13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
