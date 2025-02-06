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


# kernel path: inductor_cache/x6/cx6vqb7wt5krfko2xy7lrqwdan2ybs3o3h46gqsdewl3rfswlblg.py
# Topologically Sorted Source Nodes: [e, e_1], Original ATen: [aten.add, aten.leaky_relu]
# Source node to ATen node mapping:
#   e => add
#   e_1 => gt
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_1, %permute), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add, 0), kwargs = {})
triton_poi_fused_add_leaky_relu_0 = async_compile.triton('triton_poi_fused_add_leaky_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_leaky_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eg/cegj7u6mvkniowhhvxvp3ea62orr6wmle4en3dvjxcsigl2rkett.py
# Topologically Sorted Source Nodes: [gt], Original ATen: [aten.gt]
# Source node to ATen node mapping:
#   gt => gt_1
# Graph fragment:
#   %gt_1 : [num_users=6] = call_function[target=torch.ops.aten.gt.Scalar](args = (%primals_4, 0), kwargs = {})
triton_poi_fused_gt_1 = async_compile.triton('triton_poi_fused_gt_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gt_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gt_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ty/cty7bzhm7q5jui6xz2iq33xlf2x2yol2i4xylfvva7gmgt6ityac.py
# Topologically Sorted Source Nodes: [e, e_1, zero_vec, attention, attention_1, e_2, e_3, attention_3, attention_4, e_4, e_5, attention_6, attention_7, e_6, e_7, attention_9, attention_10], Original ATen: [aten.add, aten.leaky_relu, aten.mul, aten.where, aten._softmax]
# Source node to ATen node mapping:
#   attention => where_1
#   attention_1 => amax
#   attention_10 => amax_3
#   attention_3 => where_4
#   attention_4 => amax_1
#   attention_6 => where_7
#   attention_7 => amax_2
#   attention_9 => where_10
#   e => add
#   e_1 => mul, where
#   e_2 => add_1
#   e_3 => mul_5, where_3
#   e_4 => add_2
#   e_5 => mul_10, where_6
#   e_6 => add_3
#   e_7 => mul_15, where_9
#   zero_vec => full_default
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_1, %permute), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 4), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add, %mul), kwargs = {})
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], -8999999815811072.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where, %full_default), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_1, [1], True), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_5, %permute_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 4), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_1, %mul_5), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where_3, %full_default), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_4, [1], True), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_9, %permute_2), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 4), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_2, %mul_10), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where_6, %full_default), kwargs = {})
#   %amax_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_7, [1], True), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_13, %permute_3), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 4), kwargs = {})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %add_3, %mul_15), kwargs = {})
#   %where_10 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where_9, %full_default), kwargs = {})
#   %amax_3 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_10, [1], True), kwargs = {})
triton_poi_fused__softmax_add_leaky_relu_mul_where_2 = async_compile.triton('triton_poi_fused__softmax_add_leaky_relu_mul_where_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i1', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*i1', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i1', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_leaky_relu_mul_where_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 40, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_leaky_relu_mul_where_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp3 = tl.load(in_ptr3 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp12 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp13 = tl.load(in_ptr3 + (1))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp22 = tl.load(in_ptr3 + (2))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp29 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp30 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp31 = tl.load(in_ptr3 + (3))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp38 = tl.load(in_ptr4 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp39 = tl.load(in_ptr5 + (x0), xmask)
    tmp40 = tl.load(in_ptr6 + (0))
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp46 = tl.load(in_ptr4 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp47 = tl.load(in_ptr6 + (1))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp54 = tl.load(in_ptr4 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp55 = tl.load(in_ptr6 + (2))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK])
    tmp62 = tl.load(in_ptr4 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp63 = tl.load(in_ptr6 + (3))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK])
    tmp70 = tl.load(in_ptr7 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp71 = tl.load(in_ptr8 + (x0), xmask)
    tmp72 = tl.load(in_ptr9 + (0))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK])
    tmp78 = tl.load(in_ptr7 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp79 = tl.load(in_ptr9 + (1))
    tmp80 = tl.broadcast_to(tmp79, [XBLOCK])
    tmp86 = tl.load(in_ptr7 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp87 = tl.load(in_ptr9 + (2))
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK])
    tmp94 = tl.load(in_ptr7 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp95 = tl.load(in_ptr9 + (3))
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK])
    tmp102 = tl.load(in_ptr10 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp103 = tl.load(in_ptr11 + (x0), xmask)
    tmp104 = tl.load(in_ptr12 + (0))
    tmp105 = tl.broadcast_to(tmp104, [XBLOCK])
    tmp110 = tl.load(in_ptr10 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp111 = tl.load(in_ptr12 + (1))
    tmp112 = tl.broadcast_to(tmp111, [XBLOCK])
    tmp118 = tl.load(in_ptr10 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp119 = tl.load(in_ptr12 + (2))
    tmp120 = tl.broadcast_to(tmp119, [XBLOCK])
    tmp126 = tl.load(in_ptr10 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp127 = tl.load(in_ptr12 + (3))
    tmp128 = tl.broadcast_to(tmp127, [XBLOCK])
    tmp5 = tmp2 + tmp4
    tmp6 = 4.0
    tmp7 = tmp5 * tmp6
    tmp8 = tl.where(tmp1, tmp5, tmp7)
    tmp9 = -8999999815811072.0
    tmp10 = tl.where(tmp0, tmp8, tmp9)
    tmp15 = tmp2 + tmp14
    tmp16 = tmp15 * tmp6
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.where(tmp11, tmp17, tmp9)
    tmp19 = triton_helpers.maximum(tmp10, tmp18)
    tmp24 = tmp2 + tmp23
    tmp25 = tmp24 * tmp6
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tl.where(tmp20, tmp26, tmp9)
    tmp28 = triton_helpers.maximum(tmp19, tmp27)
    tmp33 = tmp2 + tmp32
    tmp34 = tmp33 * tmp6
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp29, tmp35, tmp9)
    tmp37 = triton_helpers.maximum(tmp28, tmp36)
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42 * tmp6
    tmp44 = tl.where(tmp38, tmp42, tmp43)
    tmp45 = tl.where(tmp0, tmp44, tmp9)
    tmp49 = tmp39 + tmp48
    tmp50 = tmp49 * tmp6
    tmp51 = tl.where(tmp46, tmp49, tmp50)
    tmp52 = tl.where(tmp11, tmp51, tmp9)
    tmp53 = triton_helpers.maximum(tmp45, tmp52)
    tmp57 = tmp39 + tmp56
    tmp58 = tmp57 * tmp6
    tmp59 = tl.where(tmp54, tmp57, tmp58)
    tmp60 = tl.where(tmp20, tmp59, tmp9)
    tmp61 = triton_helpers.maximum(tmp53, tmp60)
    tmp65 = tmp39 + tmp64
    tmp66 = tmp65 * tmp6
    tmp67 = tl.where(tmp62, tmp65, tmp66)
    tmp68 = tl.where(tmp29, tmp67, tmp9)
    tmp69 = triton_helpers.maximum(tmp61, tmp68)
    tmp74 = tmp71 + tmp73
    tmp75 = tmp74 * tmp6
    tmp76 = tl.where(tmp70, tmp74, tmp75)
    tmp77 = tl.where(tmp0, tmp76, tmp9)
    tmp81 = tmp71 + tmp80
    tmp82 = tmp81 * tmp6
    tmp83 = tl.where(tmp78, tmp81, tmp82)
    tmp84 = tl.where(tmp11, tmp83, tmp9)
    tmp85 = triton_helpers.maximum(tmp77, tmp84)
    tmp89 = tmp71 + tmp88
    tmp90 = tmp89 * tmp6
    tmp91 = tl.where(tmp86, tmp89, tmp90)
    tmp92 = tl.where(tmp20, tmp91, tmp9)
    tmp93 = triton_helpers.maximum(tmp85, tmp92)
    tmp97 = tmp71 + tmp96
    tmp98 = tmp97 * tmp6
    tmp99 = tl.where(tmp94, tmp97, tmp98)
    tmp100 = tl.where(tmp29, tmp99, tmp9)
    tmp101 = triton_helpers.maximum(tmp93, tmp100)
    tmp106 = tmp103 + tmp105
    tmp107 = tmp106 * tmp6
    tmp108 = tl.where(tmp102, tmp106, tmp107)
    tmp109 = tl.where(tmp0, tmp108, tmp9)
    tmp113 = tmp103 + tmp112
    tmp114 = tmp113 * tmp6
    tmp115 = tl.where(tmp110, tmp113, tmp114)
    tmp116 = tl.where(tmp11, tmp115, tmp9)
    tmp117 = triton_helpers.maximum(tmp109, tmp116)
    tmp121 = tmp103 + tmp120
    tmp122 = tmp121 * tmp6
    tmp123 = tl.where(tmp118, tmp121, tmp122)
    tmp124 = tl.where(tmp20, tmp123, tmp9)
    tmp125 = triton_helpers.maximum(tmp117, tmp124)
    tmp129 = tmp103 + tmp128
    tmp130 = tmp129 * tmp6
    tmp131 = tl.where(tmp126, tmp129, tmp130)
    tmp132 = tl.where(tmp29, tmp131, tmp9)
    tmp133 = triton_helpers.maximum(tmp125, tmp132)
    tl.store(out_ptr0 + (x0), tmp37, xmask)
    tl.store(out_ptr1 + (x0), tmp69, xmask)
    tl.store(out_ptr2 + (x0), tmp101, xmask)
    tl.store(out_ptr3 + (x0), tmp133, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ab/cabwjbban3lqsylvavl64rsmejmlezzoylrztl2r7jjdio3q5l37.py
# Topologically Sorted Source Nodes: [e, e_1, zero_vec, attention, attention_1, e_2, e_3, attention_3, attention_4, e_4, e_5, attention_6, attention_7, e_6, e_7, attention_9, attention_10], Original ATen: [aten.add, aten.leaky_relu, aten.mul, aten.where, aten._softmax]
# Source node to ATen node mapping:
#   attention => where_1
#   attention_1 => exp, sub
#   attention_10 => exp_3, sub_3
#   attention_3 => where_4
#   attention_4 => exp_1, sub_1
#   attention_6 => where_7
#   attention_7 => exp_2, sub_2
#   attention_9 => where_10
#   e => add
#   e_1 => mul, where
#   e_2 => add_1
#   e_3 => mul_5, where_3
#   e_4 => add_2
#   e_5 => mul_10, where_6
#   e_6 => add_3
#   e_7 => mul_15, where_9
#   zero_vec => full_default
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_1, %permute), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 4), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add, %mul), kwargs = {})
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], -8999999815811072.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where, %full_default), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_5, %permute_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 4), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_1, %mul_5), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where_3, %full_default), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_4, %amax_1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_9, %permute_2), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 4), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_2, %mul_10), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where_6, %full_default), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_7, %amax_2), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_13, %permute_3), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 4), kwargs = {})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %add_3, %mul_15), kwargs = {})
#   %where_10 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where_9, %full_default), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_10, %amax_3), kwargs = {})
#   %exp_3 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
triton_poi_fused__softmax_add_leaky_relu_mul_where_3 = async_compile.triton('triton_poi_fused__softmax_add_leaky_relu_mul_where_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i1', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*i1', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*i1', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_leaky_relu_mul_where_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_leaky_relu_mul_where_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask).to(tl.int1)
    tmp14 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x2), xmask).to(tl.int1)
    tmp24 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr13 + (x2), xmask).to(tl.int1)
    tmp34 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr15 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr16 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = 4.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.where(tmp1, tmp4, tmp6)
    tmp8 = -8999999815811072.0
    tmp9 = tl.where(tmp0, tmp7, tmp8)
    tmp11 = tmp9 - tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 * tmp5
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp0, tmp18, tmp8)
    tmp21 = tmp19 - tmp20
    tmp22 = tl_math.exp(tmp21)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26 * tmp5
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp0, tmp28, tmp8)
    tmp31 = tmp29 - tmp30
    tmp32 = tl_math.exp(tmp31)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36 * tmp5
    tmp38 = tl.where(tmp33, tmp36, tmp37)
    tmp39 = tl.where(tmp0, tmp38, tmp8)
    tmp41 = tmp39 - tmp40
    tmp42 = tl_math.exp(tmp41)
    tl.store(out_ptr0 + (x2), tmp12, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
    tl.store(out_ptr2 + (x2), tmp32, xmask)
    tl.store(out_ptr3 + (x2), tmp42, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdxeercglymjqip4jf3xl5kzkf73hf4shg7i7tr4dl3lod6ymhsr.py
# Topologically Sorted Source Nodes: [attention_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attention_1 => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_4 = async_compile.triton('triton_poi_fused__softmax_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yu/cyul7x6sjt4tobsccr33yi6ethslfngbrkmuisvf44op32j2sidl.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_1 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%where_2, %where_5, %where_8, %where_11], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 1.0
    tmp9 = tmp5 * tmp8
    tmp10 = libdevice.expm1(tmp9)
    tmp11 = tmp10 * tmp8
    tmp12 = tl.where(tmp7, tmp9, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 8, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr1 + (4*x1 + ((-4) + x0)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = 0.0
    tmp21 = tmp19 > tmp20
    tmp22 = 1.0
    tmp23 = tmp19 * tmp22
    tmp24 = libdevice.expm1(tmp23)
    tmp25 = tmp24 * tmp22
    tmp26 = tl.where(tmp21, tmp23, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp18, tmp26, tmp27)
    tmp29 = tmp0 >= tmp16
    tmp30 = tl.full([1], 12, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tl.load(in_ptr2 + (4*x1 + ((-8) + x0)), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = 0.0
    tmp35 = tmp33 > tmp34
    tmp36 = 1.0
    tmp37 = tmp33 * tmp36
    tmp38 = libdevice.expm1(tmp37)
    tmp39 = tmp38 * tmp36
    tmp40 = tl.where(tmp35, tmp37, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp32, tmp40, tmp41)
    tmp43 = tmp0 >= tmp30
    tmp44 = tl.full([1], 16, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tl.load(in_ptr3 + (4*x1 + ((-12) + x0)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = 0.0
    tmp48 = tmp46 > tmp47
    tmp49 = 1.0
    tmp50 = tmp46 * tmp49
    tmp51 = libdevice.expm1(tmp50)
    tmp52 = tmp51 * tmp49
    tmp53 = tl.where(tmp48, tmp50, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp43, tmp53, tmp54)
    tmp56 = tl.where(tmp32, tmp42, tmp55)
    tmp57 = tl.where(tmp18, tmp28, tmp56)
    tmp58 = tl.where(tmp4, tmp14, tmp57)
    tl.store(out_ptr0 + (x2), tmp58, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ax/caxd64awzyge6kequ46a2omru5b7unebhr53vylwesedyojm4px5.py
# Topologically Sorted Source Nodes: [zero_vec, e_8, e_9, attention_12, attention_13], Original ATen: [aten.mul, aten.add, aten.leaky_relu, aten.where, aten._softmax]
# Source node to ATen node mapping:
#   attention_12 => where_13
#   attention_13 => amax_4
#   e_8 => add_4
#   e_9 => mul_20, where_12
#   zero_vec => full_default
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], -8999999815811072.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_17, %permute_4), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 4), kwargs = {})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %add_4, %mul_20), kwargs = {})
#   %where_13 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where_12, %full_default), kwargs = {})
#   %amax_4 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_13, [1], True), kwargs = {})
triton_poi_fused__softmax_add_leaky_relu_mul_where_6 = async_compile.triton('triton_poi_fused__softmax_add_leaky_relu_mul_where_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_leaky_relu_mul_where_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_leaky_relu_mul_where_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp3 = tl.load(in_ptr3 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp12 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp13 = tl.load(in_ptr3 + (1))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp22 = tl.load(in_ptr3 + (2))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp29 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp30 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp31 = tl.load(in_ptr3 + (3))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp5 = tmp2 + tmp4
    tmp6 = 4.0
    tmp7 = tmp5 * tmp6
    tmp8 = tl.where(tmp1, tmp5, tmp7)
    tmp9 = -8999999815811072.0
    tmp10 = tl.where(tmp0, tmp8, tmp9)
    tmp15 = tmp2 + tmp14
    tmp16 = tmp15 * tmp6
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.where(tmp11, tmp17, tmp9)
    tmp19 = triton_helpers.maximum(tmp10, tmp18)
    tmp24 = tmp2 + tmp23
    tmp25 = tmp24 * tmp6
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tl.where(tmp20, tmp26, tmp9)
    tmp28 = triton_helpers.maximum(tmp19, tmp27)
    tmp33 = tmp2 + tmp32
    tmp34 = tmp33 * tmp6
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp29, tmp35, tmp9)
    tmp37 = triton_helpers.maximum(tmp28, tmp36)
    tl.store(out_ptr0 + (x0), tmp37, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ei/ceis6itx5wikrl2qyqdpt2bw65jrs4ffxujsa2qmzo7fgux3pdye.py
# Topologically Sorted Source Nodes: [zero_vec, e_8, e_9, attention_12, attention_13], Original ATen: [aten.mul, aten.add, aten.leaky_relu, aten.where, aten._softmax]
# Source node to ATen node mapping:
#   attention_12 => where_13
#   attention_13 => exp_4, sub_4
#   e_8 => add_4
#   e_9 => mul_20, where_12
#   zero_vec => full_default
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], -8999999815811072.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_17, %permute_4), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 4), kwargs = {})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %add_4, %mul_20), kwargs = {})
#   %where_13 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where_12, %full_default), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_13, %amax_4), kwargs = {})
#   %exp_4 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
triton_poi_fused__softmax_add_leaky_relu_mul_where_7 = async_compile.triton('triton_poi_fused__softmax_add_leaky_relu_mul_where_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_leaky_relu_mul_where_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_leaky_relu_mul_where_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = 4.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.where(tmp1, tmp4, tmp6)
    tmp8 = -8999999815811072.0
    tmp9 = tl.where(tmp0, tmp7, tmp8)
    tmp11 = tmp9 - tmp10
    tmp12 = tl_math.exp(tmp11)
    tl.store(out_ptr0 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vq/cvqauxp4bb5iicpkmaj3l5ndvozal6rmzihtl4k3xljralh7ozg3.py
# Topologically Sorted Source Nodes: [x_3, log_softmax], Original ATen: [aten.elu, aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => amax_5, sub_5
#   x_3 => expm1_4, gt_14, mul_22, mul_24, where_14
# Graph fragment:
#   %gt_14 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mm_19, 0), kwargs = {})
#   %mul_22 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_19, 1.0), kwargs = {})
#   %expm1_4 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_22,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_4, 1.0), kwargs = {})
#   %where_14 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %mul_22, %mul_24), kwargs = {})
#   %amax_5 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_14, [1], True), kwargs = {})
#   %sub_5 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_14, %amax_5), kwargs = {})
triton_poi_fused__log_softmax_elu_8 = async_compile.triton('triton_poi_fused__log_softmax_elu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_elu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_elu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp8 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tmp9 = tmp8 > tmp1
    tmp10 = tmp8 * tmp3
    tmp11 = libdevice.expm1(tmp10)
    tmp12 = tmp11 * tmp3
    tmp13 = tl.where(tmp9, tmp10, tmp12)
    tmp15 = tmp14 > tmp1
    tmp16 = tmp14 * tmp3
    tmp17 = libdevice.expm1(tmp16)
    tmp18 = tmp17 * tmp3
    tmp19 = tl.where(tmp15, tmp16, tmp18)
    tmp20 = triton_helpers.maximum(tmp13, tmp19)
    tmp22 = tmp21 > tmp1
    tmp23 = tmp21 * tmp3
    tmp24 = libdevice.expm1(tmp23)
    tmp25 = tmp24 * tmp3
    tmp26 = tl.where(tmp22, tmp23, tmp25)
    tmp27 = triton_helpers.maximum(tmp20, tmp26)
    tmp29 = tmp28 > tmp1
    tmp30 = tmp28 * tmp3
    tmp31 = libdevice.expm1(tmp30)
    tmp32 = tmp31 * tmp3
    tmp33 = tl.where(tmp29, tmp30, tmp32)
    tmp34 = triton_helpers.maximum(tmp27, tmp33)
    tmp35 = tmp7 - tmp34
    tl.store(out_ptr0 + (x2), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ie/ciekp4aur2xezusxmumi6euivtpwjsp6qxyqjawp4b5aqwpj5u4g.py
# Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => exp_5, log, sub_6, sum_6
# Graph fragment:
#   %exp_5 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_5,), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_5, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_6,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_5, %log), kwargs = {})
triton_poi_fused__log_softmax_9 = async_compile.triton('triton_poi_fused__log_softmax_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl_math.exp(tmp1)
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp2 + tmp4
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp5 + tmp7
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = tl_math.log(tmp11)
    tmp13 = tmp0 - tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (8, 1), (1, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, 4), (4, 1))
    assert_size_stride(primals_6, (8, 1), (1, 1))
    assert_size_stride(primals_7, (4, 4), (4, 1))
    assert_size_stride(primals_8, (8, 1), (1, 1))
    assert_size_stride(primals_9, (4, 4), (4, 1))
    assert_size_stride(primals_10, (8, 1), (1, 1))
    assert_size_stride(primals_11, (16, 4), (4, 1))
    assert_size_stride(primals_12, (8, 1), (1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh], Original ATen: [aten.mm]
        extern_kernels.mm(primals_1, primals_2, out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh1], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (4, 1), (1, 1), 0), out=buf1)
        buf2 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh2], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (4, 1), (1, 1), 4), out=buf2)
        buf3 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [e, e_1], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_0.run(buf1, buf2, buf3, 16, grid=grid(16), stream=stream0)
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [gt], Original ATen: [aten.gt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gt_1.run(primals_4, buf4, 16, grid=grid(16), stream=stream0)
        del primals_4
        buf9 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh_1], Original ATen: [aten.mm]
        extern_kernels.mm(primals_1, primals_5, out=buf9)
        del primals_5
        buf10 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh1_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, reinterpret_tensor(primals_6, (4, 1), (1, 1), 0), out=buf10)
        buf11 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh2_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, reinterpret_tensor(primals_6, (4, 1), (1, 1), 4), out=buf11)
        buf12 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [e_2, e_3], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_0.run(buf10, buf11, buf12, 16, grid=grid(16), stream=stream0)
        buf17 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh_2], Original ATen: [aten.mm]
        extern_kernels.mm(primals_1, primals_7, out=buf17)
        del primals_7
        buf18 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh1_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(primals_8, (4, 1), (1, 1), 0), out=buf18)
        buf19 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh2_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(primals_8, (4, 1), (1, 1), 4), out=buf19)
        buf20 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [e_4, e_5], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_0.run(buf18, buf19, buf20, 16, grid=grid(16), stream=stream0)
        buf25 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh_3], Original ATen: [aten.mm]
        extern_kernels.mm(primals_1, primals_9, out=buf25)
        del primals_9
        buf26 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh1_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, reinterpret_tensor(primals_10, (4, 1), (1, 1), 0), out=buf26)
        buf27 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh2_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, reinterpret_tensor(primals_10, (4, 1), (1, 1), 4), out=buf27)
        buf28 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [e_6, e_7], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_0.run(buf26, buf27, buf28, 16, grid=grid(16), stream=stream0)
        buf5 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf13 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf21 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf29 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [e, e_1, zero_vec, attention, attention_1, e_2, e_3, attention_3, attention_4, e_4, e_5, attention_6, attention_7, e_6, e_7, attention_9, attention_10], Original ATen: [aten.add, aten.leaky_relu, aten.mul, aten.where, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_leaky_relu_mul_where_2.run(buf4, buf3, buf1, buf2, buf12, buf10, buf11, buf20, buf18, buf19, buf28, buf26, buf27, buf5, buf13, buf21, buf29, 4, grid=grid(4), stream=stream0)
        buf6 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf14 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf22 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf30 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [e, e_1, zero_vec, attention, attention_1, e_2, e_3, attention_3, attention_4, e_4, e_5, attention_6, attention_7, e_6, e_7, attention_9, attention_10], Original ATen: [aten.add, aten.leaky_relu, aten.mul, aten.where, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_leaky_relu_mul_where_3.run(buf4, buf3, buf1, buf2, buf5, buf12, buf10, buf11, buf13, buf20, buf18, buf19, buf21, buf28, buf26, buf27, buf29, buf6, buf14, buf22, buf30, 16, grid=grid(16), stream=stream0)
        del buf1
        del buf10
        del buf11
        del buf13
        del buf18
        del buf19
        del buf2
        del buf21
        del buf26
        buf7 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf6, buf7, 16, grid=grid(16), stream=stream0)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [h_prime], Original ATen: [aten.mm]
        extern_kernels.mm(buf7, buf0, out=buf8)
        buf15 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf14, buf15, 16, grid=grid(16), stream=stream0)
        buf16 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [h_prime_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf15, buf9, out=buf16)
        buf23 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_7], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf22, buf23, 16, grid=grid(16), stream=stream0)
        buf24 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [h_prime_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf23, buf17, out=buf24)
        buf31 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_10], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf30, buf31, 16, grid=grid(16), stream=stream0)
        buf32 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [h_prime_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, buf25, out=buf32)
        buf33 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf8, buf16, buf24, buf32, buf33, 64, grid=grid(64), stream=stream0)
        buf34 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf33, primals_11, out=buf34)
        buf35 = reinterpret_tensor(buf5, (4, 1), (1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [Wh1_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_12, (4, 1), (1, 1), 0), out=buf35)
        buf36 = reinterpret_tensor(buf29, (4, 1), (1, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [Wh2_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_12, (4, 1), (1, 1), 4), out=buf36)
        buf37 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [e_8, e_9], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_0.run(buf35, buf36, buf37, 16, grid=grid(16), stream=stream0)
        buf38 = reinterpret_tensor(buf27, (4, 1), (1, 4), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [zero_vec, e_8, e_9, attention_12, attention_13], Original ATen: [aten.mul, aten.add, aten.leaky_relu, aten.where, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_leaky_relu_mul_where_6.run(buf4, buf37, buf35, buf36, buf38, 4, grid=grid(4), stream=stream0)
        buf39 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [zero_vec, e_8, e_9, attention_12, attention_13], Original ATen: [aten.mul, aten.add, aten.leaky_relu, aten.where, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_leaky_relu_mul_where_7.run(buf4, buf37, buf35, buf36, buf38, buf39, 16, grid=grid(16), stream=stream0)
        del buf35
        del buf36
        del buf38
        buf40 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_13], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf39, buf40, 16, grid=grid(16), stream=stream0)
        buf41 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [h_prime_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, buf34, out=buf41)
        buf42 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, log_softmax], Original ATen: [aten.elu, aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_elu_8.run(buf41, buf42, 16, grid=grid(16), stream=stream0)
        buf43 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_9.run(buf42, buf43, 16, grid=grid(16), stream=stream0)
        del buf42
    return (buf43, buf3, buf4, buf7, buf8, buf12, buf15, buf16, buf20, buf23, buf24, buf28, buf31, buf32, buf37, buf40, buf41, buf43, reinterpret_tensor(buf34, (4, 4), (1, 4), 0), reinterpret_tensor(primals_12, (1, 4), (1, 1), 4), reinterpret_tensor(primals_12, (1, 4), (1, 1), 0), reinterpret_tensor(buf33, (16, 4), (1, 16), 0), reinterpret_tensor(primals_11, (4, 16), (1, 4), 0), reinterpret_tensor(buf25, (4, 4), (1, 4), 0), reinterpret_tensor(primals_10, (1, 4), (1, 1), 4), reinterpret_tensor(primals_10, (1, 4), (1, 1), 0), reinterpret_tensor(primals_1, (4, 4), (1, 4), 0), reinterpret_tensor(buf17, (4, 4), (1, 4), 0), reinterpret_tensor(primals_8, (1, 4), (1, 1), 4), reinterpret_tensor(primals_8, (1, 4), (1, 1), 0), reinterpret_tensor(buf9, (4, 4), (1, 4), 0), reinterpret_tensor(primals_6, (1, 4), (1, 1), 4), reinterpret_tensor(primals_6, (1, 4), (1, 1), 0), reinterpret_tensor(buf0, (4, 4), (1, 4), 0), reinterpret_tensor(primals_3, (1, 4), (1, 1), 4), reinterpret_tensor(primals_3, (1, 4), (1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
