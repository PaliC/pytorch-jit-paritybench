# AOT ID: ['2_inference']
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


# kernel path: inductor_cache/x4/cx4amq2gg54p3kg5rjqizz3crjj4hihp46wbhbvxzhbzb7etfdh3.py
# Topologically Sorted Source Nodes: [pred], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   pred => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%arg0_1, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_0 = async_compile.triton('triton_poi_fused__softmax_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/se/cseewhmhjmmcjuqbcd4clkf34tqi36mmu3r2bj4jgbpiovepmyqv.py
# Topologically Sorted Source Nodes: [pred], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   pred => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_1 = async_compile.triton('triton_poi_fused__softmax_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/lt/cltjlxunm2v4peq6scn55z5ybhhqexgaz5q6bjcbp6kqrtpabits.py
# Topologically Sorted Source Nodes: [mul, ne, valid_mask, valid_mask_1, mul_1, sum_1, pow_1, pow_2, add_1, sum_2, mul_3, valid_mask_2, mul_4, sum_3, pow_3, pow_4, add_5, sum_4, mul_6, valid_mask_3, mul_7, sum_5, pow_5, pow_6, add_8, sum_6, mul_9, valid_mask_4, mul_10, sum_7, pow_7, pow_8, add_11, sum_8], Original ATen: [aten.mul, aten.ne, aten._to_copy, aten.view, aten.sum, aten.pow, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_11 => add_13
#   add_5 => add_5
#   add_8 => add_9
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_9 => mul_9
#   ne => ne
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_6 => pow_6
#   pow_7 => pow_7
#   pow_8 => pow_8
#   sum_1 => sum_2
#   sum_2 => sum_3
#   sum_3 => sum_4
#   sum_4 => sum_5
#   sum_5 => sum_6
#   sum_6 => sum_7
#   sum_7 => sum_8
#   sum_8 => sum_9
#   valid_mask => convert_element_type_2
#   valid_mask_1 => view_2
#   valid_mask_2 => view_5
#   valid_mask_3 => view_8
#   valid_mask_4 => view_11
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %select_1), kwargs = {})
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%arg1_1, 255), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.int64), kwargs = {})
#   %view_2 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_2, [4, -1]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %view_2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view, 2), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_1, 2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_1, [1]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %select_3), kwargs = {})
#   %view_5 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_2, [4, -1]), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %view_5), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [1]), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_3, 2), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_3, 2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_3, %pow_4), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_5, [1]), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %select_5), kwargs = {})
#   %view_8 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_2, [4, -1]), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %view_8), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_7, [1]), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_6, 2), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_5, 2), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_5, %pow_6), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_9, [1]), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %select_7), kwargs = {})
#   %view_11 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_2, [4, -1]), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %view_11), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_10, [1]), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_9, 2), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_7, 2), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_7, %pow_8), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_13, [1]), kwargs = {})
triton_poi_fused__to_copy_add_mul_ne_pow_sum_view_2 = async_compile.triton('triton_poi_fused__to_copy_add_mul_ne_pow_sum_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_ne_pow_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_ne_pow_sum_view_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.int64)
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = tl.full([1], 3, tl.int64)
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp6 == tmp3
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp0 * tmp9
    tmp11 = 255.0
    tmp12 = tmp1 != tmp11
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp10 * tmp14
    tmp17 = tmp16.to(tl.int64)
    tmp18 = triton_helpers.maximum(tmp17, tmp3)
    tmp19 = triton_helpers.minimum(tmp18, tmp5)
    tmp20 = tmp19 == tmp3
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp0 * tmp22
    tmp24 = tmp16 != tmp11
    tmp25 = tmp24.to(tl.int64)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp23 * tmp26
    tmp28 = tmp15 + tmp27
    tmp30 = tmp29.to(tl.int64)
    tmp31 = triton_helpers.maximum(tmp30, tmp3)
    tmp32 = triton_helpers.minimum(tmp31, tmp5)
    tmp33 = tmp32 == tmp3
    tmp34 = tmp33.to(tl.int64)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp0 * tmp35
    tmp37 = tmp29 != tmp11
    tmp38 = tmp37.to(tl.int64)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 * tmp39
    tmp41 = tmp28 + tmp40
    tmp43 = tmp42.to(tl.int64)
    tmp44 = triton_helpers.maximum(tmp43, tmp3)
    tmp45 = triton_helpers.minimum(tmp44, tmp5)
    tmp46 = tmp45 == tmp3
    tmp47 = tmp46.to(tl.int64)
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp0 * tmp48
    tmp50 = tmp42 != tmp11
    tmp51 = tmp50.to(tl.int64)
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp49 * tmp52
    tmp54 = tmp41 + tmp53
    tmp55 = tmp0 * tmp0
    tmp56 = tmp8 * tmp8
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp55 + tmp57
    tmp59 = tmp21 * tmp21
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp55 + tmp60
    tmp62 = tmp58 + tmp61
    tmp63 = tmp34 * tmp34
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp55 + tmp64
    tmp66 = tmp62 + tmp65
    tmp67 = tmp47 * tmp47
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp55 + tmp68
    tmp70 = tmp66 + tmp69
    tmp72 = tl.full([1], 1, tl.int64)
    tmp73 = tmp6 == tmp72
    tmp74 = tmp73.to(tl.int64)
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tmp71 * tmp75
    tmp77 = tmp76 * tmp14
    tmp78 = tmp19 == tmp72
    tmp79 = tmp78.to(tl.int64)
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp71 * tmp80
    tmp82 = tmp81 * tmp26
    tmp83 = tmp77 + tmp82
    tmp84 = tmp32 == tmp72
    tmp85 = tmp84.to(tl.int64)
    tmp86 = tmp85.to(tl.float32)
    tmp87 = tmp71 * tmp86
    tmp88 = tmp87 * tmp39
    tmp89 = tmp83 + tmp88
    tmp90 = tmp45 == tmp72
    tmp91 = tmp90.to(tl.int64)
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp71 * tmp92
    tmp94 = tmp93 * tmp52
    tmp95 = tmp89 + tmp94
    tmp96 = tmp71 * tmp71
    tmp97 = tmp74 * tmp74
    tmp98 = tmp97.to(tl.float32)
    tmp99 = tmp96 + tmp98
    tmp100 = tmp79 * tmp79
    tmp101 = tmp100.to(tl.float32)
    tmp102 = tmp96 + tmp101
    tmp103 = tmp99 + tmp102
    tmp104 = tmp85 * tmp85
    tmp105 = tmp104.to(tl.float32)
    tmp106 = tmp96 + tmp105
    tmp107 = tmp103 + tmp106
    tmp108 = tmp91 * tmp91
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tmp96 + tmp109
    tmp111 = tmp107 + tmp110
    tmp113 = tl.full([1], 2, tl.int64)
    tmp114 = tmp6 == tmp113
    tmp115 = tmp114.to(tl.int64)
    tmp116 = tmp115.to(tl.float32)
    tmp117 = tmp112 * tmp116
    tmp118 = tmp117 * tmp14
    tmp119 = tmp19 == tmp113
    tmp120 = tmp119.to(tl.int64)
    tmp121 = tmp120.to(tl.float32)
    tmp122 = tmp112 * tmp121
    tmp123 = tmp122 * tmp26
    tmp124 = tmp118 + tmp123
    tmp125 = tmp32 == tmp113
    tmp126 = tmp125.to(tl.int64)
    tmp127 = tmp126.to(tl.float32)
    tmp128 = tmp112 * tmp127
    tmp129 = tmp128 * tmp39
    tmp130 = tmp124 + tmp129
    tmp131 = tmp45 == tmp113
    tmp132 = tmp131.to(tl.int64)
    tmp133 = tmp132.to(tl.float32)
    tmp134 = tmp112 * tmp133
    tmp135 = tmp134 * tmp52
    tmp136 = tmp130 + tmp135
    tmp137 = tmp112 * tmp112
    tmp138 = tmp115 * tmp115
    tmp139 = tmp138.to(tl.float32)
    tmp140 = tmp137 + tmp139
    tmp141 = tmp120 * tmp120
    tmp142 = tmp141.to(tl.float32)
    tmp143 = tmp137 + tmp142
    tmp144 = tmp140 + tmp143
    tmp145 = tmp126 * tmp126
    tmp146 = tmp145.to(tl.float32)
    tmp147 = tmp137 + tmp146
    tmp148 = tmp144 + tmp147
    tmp149 = tmp132 * tmp132
    tmp150 = tmp149.to(tl.float32)
    tmp151 = tmp137 + tmp150
    tmp152 = tmp148 + tmp151
    tmp154 = tmp6 == tmp5
    tmp155 = tmp154.to(tl.int64)
    tmp156 = tmp155.to(tl.float32)
    tmp157 = tmp153 * tmp156
    tmp158 = tmp157 * tmp14
    tmp159 = tmp19 == tmp5
    tmp160 = tmp159.to(tl.int64)
    tmp161 = tmp160.to(tl.float32)
    tmp162 = tmp153 * tmp161
    tmp163 = tmp162 * tmp26
    tmp164 = tmp158 + tmp163
    tmp165 = tmp32 == tmp5
    tmp166 = tmp165.to(tl.int64)
    tmp167 = tmp166.to(tl.float32)
    tmp168 = tmp153 * tmp167
    tmp169 = tmp168 * tmp39
    tmp170 = tmp164 + tmp169
    tmp171 = tmp45 == tmp5
    tmp172 = tmp171.to(tl.int64)
    tmp173 = tmp172.to(tl.float32)
    tmp174 = tmp153 * tmp173
    tmp175 = tmp174 * tmp52
    tmp176 = tmp170 + tmp175
    tmp177 = tmp153 * tmp153
    tmp178 = tmp155 * tmp155
    tmp179 = tmp178.to(tl.float32)
    tmp180 = tmp177 + tmp179
    tmp181 = tmp160 * tmp160
    tmp182 = tmp181.to(tl.float32)
    tmp183 = tmp177 + tmp182
    tmp184 = tmp180 + tmp183
    tmp185 = tmp166 * tmp166
    tmp186 = tmp185.to(tl.float32)
    tmp187 = tmp177 + tmp186
    tmp188 = tmp184 + tmp187
    tmp189 = tmp172 * tmp172
    tmp190 = tmp189.to(tl.float32)
    tmp191 = tmp177 + tmp190
    tmp192 = tmp188 + tmp191
    tl.store(out_ptr0 + (x0), tmp54, xmask)
    tl.store(out_ptr1 + (x0), tmp70, xmask)
    tl.store(out_ptr2 + (x0), tmp95, xmask)
    tl.store(out_ptr3 + (x0), tmp111, xmask)
    tl.store(out_ptr4 + (x0), tmp136, xmask)
    tl.store(out_ptr5 + (x0), tmp152, xmask)
    tl.store(out_ptr6 + (x0), tmp176, xmask)
    tl.store(out_ptr7 + (x0), tmp192, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f2/cf27w6qddem43ek4xynnfn4yyvl5j4yktj7pj5ld574shtoh22q5.py
# Topologically Sorted Source Nodes: [mul_2, num, den, truediv, loss, loss_1, total_loss, mul_5, num_1, den_1, truediv_1, loss_2, loss_3, total_loss_1, mul_8, num_2, den_2, truediv_2, loss_4, loss_5, total_loss_2, mul_11, num_3, den_3, truediv_3, loss_6, loss_7, total_loss_3, loss_8, loss_9, loss_10], Original ATen: [aten.mul, aten.add, aten.div, aten.rsub, aten.mean]
# Source node to ATen node mapping:
#   den => add_2
#   den_1 => add_6
#   den_2 => add_10
#   den_3 => add_14
#   loss => sub_1
#   loss_1 => mean
#   loss_10 => mul_12
#   loss_2 => sub_2
#   loss_3 => mean_1
#   loss_4 => sub_3
#   loss_5 => mean_2
#   loss_6 => sub_4
#   loss_7 => mean_3
#   loss_8 => div_5
#   loss_9 => mean_4
#   mul_11 => mul_11
#   mul_2 => mul_2
#   mul_5 => mul_5
#   mul_8 => mul_8
#   num => add
#   num_1 => add_4
#   num_2 => add_8
#   num_3 => add_12
#   total_loss => add_3
#   total_loss_1 => add_7
#   total_loss_2 => add_11
#   total_loss_3 => add_15
#   truediv => div_1
#   truediv_1 => div_2
#   truediv_2 => div_3
#   truediv_3 => div_4
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, 2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, 1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add, %add_2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_1,), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_4, 2), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, 1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_5, 1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, %add_6), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div_2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_2,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mean_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_6, 2), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, 1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_7, 1), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_8, %add_10), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div_3), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_3,), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %mean_2), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_8, 2), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, 1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_9, 1), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_12, %add_14), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div_4), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_4,), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mean_3), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_15, 4), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%div_5,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_4, 1.0), kwargs = {})
triton_poi_fused_add_div_mean_mul_rsub_3 = async_compile.triton('triton_poi_fused_add_div_mean_mul_rsub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_mul_rsub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_mul_rsub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp15 = tl.load(in_ptr1 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (2))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp25 = tl.load(in_ptr1 + (2))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (3))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp35 = tl.load(in_ptr1 + (3))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp45 = tl.load(in_ptr2 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp49 = tl.load(in_ptr3 + (0))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp54 = tl.load(in_ptr2 + (1))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp58 = tl.load(in_ptr3 + (1))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp64 = tl.load(in_ptr2 + (2))
    tmp65 = tl.broadcast_to(tmp64, [XBLOCK])
    tmp68 = tl.load(in_ptr3 + (2))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
    tmp74 = tl.load(in_ptr2 + (3))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK])
    tmp78 = tl.load(in_ptr3 + (3))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK])
    tmp86 = tl.load(in_ptr4 + (0))
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK])
    tmp90 = tl.load(in_ptr5 + (0))
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK])
    tmp95 = tl.load(in_ptr4 + (1))
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK])
    tmp99 = tl.load(in_ptr5 + (1))
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK])
    tmp105 = tl.load(in_ptr4 + (2))
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK])
    tmp109 = tl.load(in_ptr5 + (2))
    tmp110 = tl.broadcast_to(tmp109, [XBLOCK])
    tmp115 = tl.load(in_ptr4 + (3))
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK])
    tmp119 = tl.load(in_ptr5 + (3))
    tmp120 = tl.broadcast_to(tmp119, [XBLOCK])
    tmp127 = tl.load(in_ptr6 + (0))
    tmp128 = tl.broadcast_to(tmp127, [XBLOCK])
    tmp131 = tl.load(in_ptr7 + (0))
    tmp132 = tl.broadcast_to(tmp131, [XBLOCK])
    tmp136 = tl.load(in_ptr6 + (1))
    tmp137 = tl.broadcast_to(tmp136, [XBLOCK])
    tmp140 = tl.load(in_ptr7 + (1))
    tmp141 = tl.broadcast_to(tmp140, [XBLOCK])
    tmp146 = tl.load(in_ptr6 + (2))
    tmp147 = tl.broadcast_to(tmp146, [XBLOCK])
    tmp150 = tl.load(in_ptr7 + (2))
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK])
    tmp156 = tl.load(in_ptr6 + (3))
    tmp157 = tl.broadcast_to(tmp156, [XBLOCK])
    tmp160 = tl.load(in_ptr7 + (3))
    tmp161 = tl.broadcast_to(tmp160, [XBLOCK])
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp8 = tmp7 + tmp4
    tmp9 = tmp5 / tmp8
    tmp10 = tmp4 - tmp9
    tmp13 = tmp12 * tmp2
    tmp14 = tmp13 + tmp4
    tmp17 = tmp16 + tmp4
    tmp18 = tmp14 / tmp17
    tmp19 = tmp4 - tmp18
    tmp20 = tmp10 + tmp19
    tmp23 = tmp22 * tmp2
    tmp24 = tmp23 + tmp4
    tmp27 = tmp26 + tmp4
    tmp28 = tmp24 / tmp27
    tmp29 = tmp4 - tmp28
    tmp30 = tmp20 + tmp29
    tmp33 = tmp32 * tmp2
    tmp34 = tmp33 + tmp4
    tmp37 = tmp36 + tmp4
    tmp38 = tmp34 / tmp37
    tmp39 = tmp4 - tmp38
    tmp40 = tmp30 + tmp39
    tmp41 = 4.0
    tmp42 = tmp40 / tmp41
    tmp43 = 0.0
    tmp44 = tmp42 + tmp43
    tmp47 = tmp46 * tmp2
    tmp48 = tmp47 + tmp4
    tmp51 = tmp50 + tmp4
    tmp52 = tmp48 / tmp51
    tmp53 = tmp4 - tmp52
    tmp56 = tmp55 * tmp2
    tmp57 = tmp56 + tmp4
    tmp60 = tmp59 + tmp4
    tmp61 = tmp57 / tmp60
    tmp62 = tmp4 - tmp61
    tmp63 = tmp53 + tmp62
    tmp66 = tmp65 * tmp2
    tmp67 = tmp66 + tmp4
    tmp70 = tmp69 + tmp4
    tmp71 = tmp67 / tmp70
    tmp72 = tmp4 - tmp71
    tmp73 = tmp63 + tmp72
    tmp76 = tmp75 * tmp2
    tmp77 = tmp76 + tmp4
    tmp80 = tmp79 + tmp4
    tmp81 = tmp77 / tmp80
    tmp82 = tmp4 - tmp81
    tmp83 = tmp73 + tmp82
    tmp84 = tmp83 / tmp41
    tmp85 = tmp44 + tmp84
    tmp88 = tmp87 * tmp2
    tmp89 = tmp88 + tmp4
    tmp92 = tmp91 + tmp4
    tmp93 = tmp89 / tmp92
    tmp94 = tmp4 - tmp93
    tmp97 = tmp96 * tmp2
    tmp98 = tmp97 + tmp4
    tmp101 = tmp100 + tmp4
    tmp102 = tmp98 / tmp101
    tmp103 = tmp4 - tmp102
    tmp104 = tmp94 + tmp103
    tmp107 = tmp106 * tmp2
    tmp108 = tmp107 + tmp4
    tmp111 = tmp110 + tmp4
    tmp112 = tmp108 / tmp111
    tmp113 = tmp4 - tmp112
    tmp114 = tmp104 + tmp113
    tmp117 = tmp116 * tmp2
    tmp118 = tmp117 + tmp4
    tmp121 = tmp120 + tmp4
    tmp122 = tmp118 / tmp121
    tmp123 = tmp4 - tmp122
    tmp124 = tmp114 + tmp123
    tmp125 = tmp124 / tmp41
    tmp126 = tmp85 + tmp125
    tmp129 = tmp128 * tmp2
    tmp130 = tmp129 + tmp4
    tmp133 = tmp132 + tmp4
    tmp134 = tmp130 / tmp133
    tmp135 = tmp4 - tmp134
    tmp138 = tmp137 * tmp2
    tmp139 = tmp138 + tmp4
    tmp142 = tmp141 + tmp4
    tmp143 = tmp139 / tmp142
    tmp144 = tmp4 - tmp143
    tmp145 = tmp135 + tmp144
    tmp148 = tmp147 * tmp2
    tmp149 = tmp148 + tmp4
    tmp152 = tmp151 + tmp4
    tmp153 = tmp149 / tmp152
    tmp154 = tmp4 - tmp153
    tmp155 = tmp145 + tmp154
    tmp158 = tmp157 * tmp2
    tmp159 = tmp158 + tmp4
    tmp162 = tmp161 + tmp4
    tmp163 = tmp159 / tmp162
    tmp164 = tmp4 - tmp163
    tmp165 = tmp155 + tmp164
    tmp166 = tmp165 / tmp41
    tmp167 = tmp126 + tmp166
    tmp168 = 0.25
    tmp169 = tmp167 * tmp168
    tmp170 = tmp169 / tmp4
    tmp171 = tmp170 * tmp4
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp171, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pred], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_0.run(arg0_1, buf0, 16, grid=grid(16), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pred], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf0, buf1, 16, grid=grid(16), stream=stream0)
        del buf0
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf3 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf4 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf5 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf6 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf7 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf9 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf10 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul, ne, valid_mask, valid_mask_1, mul_1, sum_1, pow_1, pow_2, add_1, sum_2, mul_3, valid_mask_2, mul_4, sum_3, pow_3, pow_4, add_5, sum_4, mul_6, valid_mask_3, mul_7, sum_5, pow_5, pow_6, add_8, sum_6, mul_9, valid_mask_4, mul_10, sum_7, pow_7, pow_8, add_11, sum_8], Original ATen: [aten.mul, aten.ne, aten._to_copy, aten.view, aten.sum, aten.pow, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mul_ne_pow_sum_view_2.run(buf1, arg1_1, buf2, buf3, buf4, buf5, buf6, buf7, buf9, buf10, 4, grid=grid(4), stream=stream0)
        del arg1_1
        del buf1
        buf8 = empty_strided_cuda((), (), torch.float32)
        buf11 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [mul_2, num, den, truediv, loss, loss_1, total_loss, mul_5, num_1, den_1, truediv_1, loss_2, loss_3, total_loss_1, mul_8, num_2, den_2, truediv_2, loss_4, loss_5, total_loss_2, mul_11, num_3, den_3, truediv_3, loss_6, loss_7, total_loss_3, loss_8, loss_9, loss_10], Original ATen: [aten.mul, aten.add, aten.div, aten.rsub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_mul_rsub_3.run(buf11, buf2, buf3, buf4, buf5, buf6, buf7, buf9, buf10, 1, grid=grid(1), stream=stream0)
        del buf10
        del buf2
        del buf3
        del buf4
        del buf5
        del buf6
        del buf7
        del buf9
    return (buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
