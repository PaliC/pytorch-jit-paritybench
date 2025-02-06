# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/6n/c6ncqz4hwm2syxmtuxbvtfm6qkadyddpqbnzp3xgffeowrffryar.py
# Topologically Sorted Source Nodes: [ref_sq, pos_sq, add, mul, sub, pos_dist_1], Original ATen: [aten.sum, aten.add, aten.mul, aten.sub, aten.neg]
# Source node to ATen node mapping:
#   add => add
#   mul => mul
#   pos_dist_1 => neg
#   pos_sq => sum_2
#   ref_sq => sum_1
#   sub => sub
# Graph fragment:
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute, [1]), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_1, [1]), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %sum_2), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mul), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%sub,), kwargs = {})
triton_poi_fused_add_mul_neg_sub_sum_0 = async_compile.triton('triton_poi_fused_add_mul_neg_sub_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_neg_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_neg_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp10 + tmp21
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 - tmp25
    tmp27 = -tmp26
    tl.store(in_out_ptr0 + (x0), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gn/cgnlsw4aonjlmwpt6f4jvve4dskvjdsg5hu4yvxeozb4mhrxkqqe.py
# Topologically Sorted Source Nodes: [add_1, mul_1, sub_1, neg_dist_1], Original ATen: [aten.add, aten.mul, aten.sub, aten.neg]
# Source node to ATen node mapping:
#   add_1 => add_1
#   mul_1 => mul_1
#   neg_dist_1 => neg_1
#   sub_1 => sub_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %mul_1), kwargs = {})
#   %neg_1 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%sub_1,), kwargs = {})
triton_poi_fused_add_mul_neg_sub_1 = async_compile.triton('triton_poi_fused_add_mul_neg_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_neg_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_neg_sub_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp10 + tmp21
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 - tmp25
    tmp27 = -tmp26
    tl.store(in_out_ptr0 + (x2), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7v/c7vrjz7lkegzcegrsm7r3ewigpgfhii6srn6vy6smvqo6eycqi3x.py
# Topologically Sorted Source Nodes: [inverse_temperature, inverse_temperature_1, neg_dist_2, max_1, neg_dist_3, logsumexp], Original ATen: [aten.exp, aten.clamp, aten.mul, aten.max, aten.sub, aten.logsumexp]
# Source node to ATen node mapping:
#   inverse_temperature => exp
#   inverse_temperature_1 => clamp_max
#   logsumexp => abs_1, amax, eq, exp_1, full_default, sub_4, sum_4, where
#   max_1 => max_1
#   neg_dist_2 => mul_3
#   neg_dist_3 => sub_3
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%primals_1,), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%exp, inf), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %clamp_max), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%mul_3, 1, True), kwargs = {})
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, %getitem), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%sub_3, [1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %where), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1]), kwargs = {})
triton_poi_fused_clamp_exp_logsumexp_max_mul_sub_2 = async_compile.triton('triton_poi_fused_clamp_exp_logsumexp_max_mul_sub_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_exp_logsumexp_max_mul_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_exp_logsumexp_max_mul_sub_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp7 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl_math.exp(tmp2)
    tmp4 = float("inf")
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp7 * tmp5
    tmp9 = triton_helpers.maximum(tmp6, tmp8)
    tmp11 = tmp10 * tmp5
    tmp12 = triton_helpers.maximum(tmp9, tmp11)
    tmp14 = tmp13 * tmp5
    tmp15 = triton_helpers.maximum(tmp12, tmp14)
    tmp16 = tmp6 - tmp15
    tmp17 = tmp8 - tmp15
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = tmp11 - tmp15
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = tmp14 - tmp15
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = tl_math.abs(tmp22)
    tmp24 = tmp23 == tmp4
    tmp25 = 0.0
    tmp26 = tl.where(tmp24, tmp25, tmp22)
    tmp27 = tmp16 - tmp26
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp17 - tmp26
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp28 + tmp30
    tmp32 = tmp19 - tmp26
    tmp33 = tl_math.exp(tmp32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp21 - tmp26
    tmp36 = tl_math.exp(tmp35)
    tmp37 = tmp34 + tmp36
    tl.store(out_ptr0 + (x0), tmp15, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
    tl.store(out_ptr2 + (x0), tmp37, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6x/c6xk6i7ggboysl7wdtlxoxnnvqix62xdad5z32zp5nex4iortdna.py
# Topologically Sorted Source Nodes: [inverse_temperature, inverse_temperature_1, pos_dist_2, pos_dist_3, neg_2, align, logsumexp, uniform, c_mean, align_corrected, uniform_corrected, add_3], Original ATen: [aten.exp, aten.clamp, aten.mul, aten.sub, aten.neg, aten.mean, aten.logsumexp, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_4
#   align => mean
#   align_corrected => sub_5
#   c_mean => mean_2
#   inverse_temperature => exp
#   inverse_temperature_1 => clamp_max
#   logsumexp => add_2, log
#   neg_2 => neg_2
#   pos_dist_2 => mul_2
#   pos_dist_3 => sub_2
#   uniform => mean_1
#   uniform_corrected => add_3
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%primals_1,), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%exp, inf), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %clamp_max), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %squeeze), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sub_2,), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%neg_2,), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_4,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze_1), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%add_2,), kwargs = {})
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%getitem,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %mean_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, %mean_2), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
triton_poi_fused_add_clamp_exp_logsumexp_mean_mul_neg_sub_3 = async_compile.triton('triton_poi_fused_add_clamp_exp_logsumexp_mean_mul_neg_sub_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_exp_logsumexp_mean_mul_neg_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_exp_logsumexp_mean_mul_neg_sub_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp14 = tl.load(in_ptr1 + (1))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (2))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (2))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (3))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp34 = tl.load(in_ptr1 + (3))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp43 = tl.load(in_ptr2 + (0))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp45 = tl.load(in_ptr3 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp50 = tl.load(in_ptr4 + (0))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp54 = tl.load(in_ptr2 + (1))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp57 = tl.load(in_ptr4 + (1))
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK])
    tmp62 = tl.load(in_ptr2 + (2))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp65 = tl.load(in_ptr4 + (2))
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp70 = tl.load(in_ptr2 + (3))
    tmp71 = tl.broadcast_to(tmp70, [XBLOCK])
    tmp73 = tl.load(in_ptr4 + (3))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK])
    tmp2 = tl_math.log(tmp1)
    tmp5 = tl_math.abs(tmp4)
    tmp6 = float("inf")
    tmp7 = tmp5 == tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp2 + tmp9
    tmp13 = tl_math.log(tmp12)
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tmp16 == tmp6
    tmp18 = tl.where(tmp17, tmp8, tmp15)
    tmp19 = tmp13 + tmp18
    tmp20 = tmp10 + tmp19
    tmp23 = tl_math.log(tmp22)
    tmp26 = tl_math.abs(tmp25)
    tmp27 = tmp26 == tmp6
    tmp28 = tl.where(tmp27, tmp8, tmp25)
    tmp29 = tmp23 + tmp28
    tmp30 = tmp20 + tmp29
    tmp33 = tl_math.log(tmp32)
    tmp36 = tl_math.abs(tmp35)
    tmp37 = tmp36 == tmp6
    tmp38 = tl.where(tmp37, tmp8, tmp35)
    tmp39 = tmp33 + tmp38
    tmp40 = tmp30 + tmp39
    tmp41 = 4.0
    tmp42 = tmp40 / tmp41
    tmp47 = tl_math.exp(tmp46)
    tmp48 = triton_helpers.minimum(tmp47, tmp6)
    tmp49 = tmp44 * tmp48
    tmp52 = tmp49 - tmp51
    tmp53 = -tmp52
    tmp56 = tmp55 * tmp48
    tmp59 = tmp56 - tmp58
    tmp60 = -tmp59
    tmp61 = tmp53 + tmp60
    tmp64 = tmp63 * tmp48
    tmp67 = tmp64 - tmp66
    tmp68 = -tmp67
    tmp69 = tmp61 + tmp68
    tmp72 = tmp71 * tmp48
    tmp75 = tmp72 - tmp74
    tmp76 = -tmp75
    tmp77 = tmp69 + tmp76
    tmp78 = tmp77 / tmp41
    tmp79 = tmp51 + tmp58
    tmp80 = tmp79 + tmp66
    tmp81 = tmp80 + tmp74
    tmp82 = tmp81 / tmp41
    tmp83 = tmp78 - tmp82
    tmp84 = tmp42 + tmp82
    tmp85 = tmp78 + tmp42
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp83, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp84, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp85, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (), ())
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pos_dist], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_2, (4, 1, 4), (4, 4, 1), 0), reinterpret_tensor(primals_3, (4, 4, 1), (4, 1, 1), 0), out=buf0)
        buf1 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [neg_dist], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_2, (1, 4, 4), (16, 4, 1), 0), reinterpret_tensor(primals_4, (1, 4, 4), (0, 1, 4), 0), out=buf1)
        buf2 = reinterpret_tensor(buf0, (4, ), (1, ), 0); del buf0  # reuse
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [ref_sq, pos_sq, add, mul, sub, pos_dist_1], Original ATen: [aten.sum, aten.add, aten.mul, aten.sub, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_neg_sub_sum_0.run(buf3, primals_2, primals_3, 4, grid=grid(4), stream=stream0)
        del primals_3
        buf4 = reinterpret_tensor(buf1, (4, 4), (4, 1), 0); del buf1  # reuse
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [add_1, mul_1, sub_1, neg_dist_1], Original ATen: [aten.add, aten.mul, aten.sub, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_neg_sub_1.run(buf5, primals_2, primals_4, 16, grid=grid(16), stream=stream0)
        del primals_2
        del primals_4
        buf6 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        buf8 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf9 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [inverse_temperature, inverse_temperature_1, neg_dist_2, max_1, neg_dist_3, logsumexp], Original ATen: [aten.exp, aten.clamp, aten.mul, aten.max, aten.sub, aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_exp_logsumexp_max_mul_sub_2.run(buf5, primals_1, buf6, buf8, buf9, 4, grid=grid(4), stream=stream0)
        buf12 = empty_strided_cuda((), (), torch.float32)
        buf13 = empty_strided_cuda((), (), torch.float32)
        buf11 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [inverse_temperature, inverse_temperature_1, pos_dist_2, pos_dist_3, neg_2, align, logsumexp, uniform, c_mean, align_corrected, uniform_corrected, add_3], Original ATen: [aten.exp, aten.clamp, aten.mul, aten.sub, aten.neg, aten.mean, aten.logsumexp, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_logsumexp_mean_mul_neg_sub_3.run(buf9, buf8, buf3, primals_1, buf6, buf12, buf13, buf11, 1, grid=grid(1), stream=stream0)
        del buf6
        del buf8
        del buf9
    return (buf11, buf12, buf13, primals_1, buf3, buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
