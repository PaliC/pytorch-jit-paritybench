# AOT ID: ['6_inference']
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


# kernel path: inductor_cache/qa/cqa5gikjzwecjn6w2tubeawoxln6frtlm5kzx5g4dnbpcxkp6les.py
# Topologically Sorted Source Nodes: [add_1, neg_dist, mul_1, sub_1], Original ATen: [aten.add, aten.div, aten.mul, aten.sub]
# Source node to ATen node mapping:
#   add_1 => add_1
#   mul_1 => mul_1
#   neg_dist => div_4
#   sub_1 => sub_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_7, 1.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %mul_1), kwargs = {})
triton_poi_fused_add_div_mul_sub_0 = async_compile.triton('triton_poi_fused_add_div_mul_sub_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp13 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 + tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 + tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 + tmp22
    tmp24 = tmp23 * tmp11
    tmp25 = tmp12 + tmp24
    tmp27 = tmp26 * tmp11
    tmp28 = 2.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 - tmp29
    tl.store(in_out_ptr0 + (x2), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m2/cm2az43dinrwkgx2vhqa6nqimenloryuki46ebrpj74wkajprhcv.py
# Topologically Sorted Source Nodes: [einsum, ref_sq, einsum_1, pos_sq, add, pos_dist, mul, sub], Original ATen: [aten.sum, aten.div, aten.add, aten.mul, aten.sub]
# Source node to ATen node mapping:
#   add => add
#   einsum => sum_1
#   einsum_1 => sum_2
#   mul => mul
#   pos_dist => div_3
#   pos_sq => div_1
#   ref_sq => div
#   sub => sub
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute, [1]), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, 1.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_1, [1]), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, 1.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %div_1), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_3, 1.0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mul), kwargs = {})
triton_poi_fused_add_div_mul_sub_sum_1 = async_compile.triton('triton_poi_fused_add_div_mul_sub_sum_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 + tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 + tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 + tmp22
    tmp24 = tmp23 * tmp11
    tmp25 = tmp12 + tmp24
    tmp27 = tmp26 * tmp11
    tmp28 = 2.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 - tmp29
    tl.store(in_out_ptr0 + (x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pq/cpqh4ydeo55hsfoeehnkouqcqlsa5fffuc2lsfoz533vxdez5j7z.py
# Topologically Sorted Source Nodes: [neg_dist_1, max_1, neg_dist_2], Original ATen: [aten.neg, aten.max, aten.sub]
# Source node to ATen node mapping:
#   max_1 => max_1
#   neg_dist_1 => neg_1
#   neg_dist_2 => sub_3
# Graph fragment:
#   %neg_1 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%sub_1,), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%neg_1, 1, True), kwargs = {})
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg_1, %getitem), kwargs = {})
triton_poi_fused_max_neg_sub_2 = async_compile.triton('triton_poi_fused_max_neg_sub_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_neg_sub_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_neg_sub_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp3 = -tmp2
    tmp5 = -tmp4
    tmp6 = triton_helpers.maximum(tmp3, tmp5)
    tmp8 = -tmp7
    tmp9 = triton_helpers.maximum(tmp6, tmp8)
    tmp11 = -tmp10
    tmp12 = triton_helpers.maximum(tmp9, tmp11)
    tmp13 = tmp1 - tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lt/cltcrfswpmsp6jperwckeoolkzlidy6pvqy4xe5lph2yhklgdu45.py
# Topologically Sorted Source Nodes: [pos_dist_1, pos_dist_2, neg_2, align, logsumexp, uniform, add_2], Original ATen: [aten.neg, aten.sub, aten.mean, aten.logsumexp, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_3
#   align => mean
#   logsumexp => abs_1, add_2, amax, eq, exp, full_default, log, sub_4, sum_4, where
#   neg_2 => neg_2
#   pos_dist_1 => neg
#   pos_dist_2 => sub_2
#   uniform => mean_1
# Graph fragment:
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sub,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %squeeze), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sub_2,), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%neg_2,), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%sub_3, [1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %where), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_4,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze_1), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%add_2,), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
triton_poi_fused_add_logsumexp_mean_neg_sub_3 = async_compile.triton('triton_poi_fused_add_logsumexp_mean_neg_sub_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_logsumexp_mean_neg_sub_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 36, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_logsumexp_mean_neg_sub_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (1))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp10 = tl.load(in_ptr1 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr1 + (3))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (1))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.load(in_ptr1 + (4))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp26 = tl.load(in_ptr1 + (5))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp30 = tl.load(in_ptr1 + (6))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp34 = tl.load(in_ptr1 + (7))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp41 = tl.load(in_ptr0 + (2))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.load(in_ptr1 + (8))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp47 = tl.load(in_ptr1 + (9))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp51 = tl.load(in_ptr1 + (10))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp55 = tl.load(in_ptr1 + (11))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK])
    tmp62 = tl.load(in_ptr0 + (3))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp65 = tl.load(in_ptr1 + (12))
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp68 = tl.load(in_ptr1 + (13))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
    tmp72 = tl.load(in_ptr1 + (14))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK])
    tmp76 = tl.load(in_ptr1 + (15))
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK])
    tmp85 = tl.load(in_ptr2 + (0))
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK])
    tmp87 = tl.load(in_ptr2 + (1))
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK])
    tmp90 = tl.load(in_ptr2 + (2))
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK])
    tmp93 = tl.load(in_ptr2 + (3))
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK])
    tmp114 = tl.load(in_ptr2 + (4))
    tmp115 = tl.broadcast_to(tmp114, [XBLOCK])
    tmp116 = tl.load(in_ptr2 + (5))
    tmp117 = tl.broadcast_to(tmp116, [XBLOCK])
    tmp119 = tl.load(in_ptr2 + (6))
    tmp120 = tl.broadcast_to(tmp119, [XBLOCK])
    tmp122 = tl.load(in_ptr2 + (7))
    tmp123 = tl.broadcast_to(tmp122, [XBLOCK])
    tmp142 = tl.load(in_ptr2 + (8))
    tmp143 = tl.broadcast_to(tmp142, [XBLOCK])
    tmp144 = tl.load(in_ptr2 + (9))
    tmp145 = tl.broadcast_to(tmp144, [XBLOCK])
    tmp147 = tl.load(in_ptr2 + (10))
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK])
    tmp150 = tl.load(in_ptr2 + (11))
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK])
    tmp170 = tl.load(in_ptr2 + (12))
    tmp171 = tl.broadcast_to(tmp170, [XBLOCK])
    tmp172 = tl.load(in_ptr2 + (13))
    tmp173 = tl.broadcast_to(tmp172, [XBLOCK])
    tmp175 = tl.load(in_ptr2 + (14))
    tmp176 = tl.broadcast_to(tmp175, [XBLOCK])
    tmp178 = tl.load(in_ptr2 + (15))
    tmp179 = tl.broadcast_to(tmp178, [XBLOCK])
    tmp2 = -tmp1
    tmp5 = -tmp4
    tmp8 = -tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp12 = -tmp11
    tmp13 = triton_helpers.maximum(tmp9, tmp12)
    tmp16 = -tmp15
    tmp17 = triton_helpers.maximum(tmp13, tmp16)
    tmp18 = tmp2 - tmp17
    tmp19 = -tmp18
    tmp22 = -tmp21
    tmp25 = -tmp24
    tmp28 = -tmp27
    tmp29 = triton_helpers.maximum(tmp25, tmp28)
    tmp32 = -tmp31
    tmp33 = triton_helpers.maximum(tmp29, tmp32)
    tmp36 = -tmp35
    tmp37 = triton_helpers.maximum(tmp33, tmp36)
    tmp38 = tmp22 - tmp37
    tmp39 = -tmp38
    tmp40 = tmp19 + tmp39
    tmp43 = -tmp42
    tmp46 = -tmp45
    tmp49 = -tmp48
    tmp50 = triton_helpers.maximum(tmp46, tmp49)
    tmp53 = -tmp52
    tmp54 = triton_helpers.maximum(tmp50, tmp53)
    tmp57 = -tmp56
    tmp58 = triton_helpers.maximum(tmp54, tmp57)
    tmp59 = tmp43 - tmp58
    tmp60 = -tmp59
    tmp61 = tmp40 + tmp60
    tmp64 = -tmp63
    tmp67 = -tmp66
    tmp70 = -tmp69
    tmp71 = triton_helpers.maximum(tmp67, tmp70)
    tmp74 = -tmp73
    tmp75 = triton_helpers.maximum(tmp71, tmp74)
    tmp78 = -tmp77
    tmp79 = triton_helpers.maximum(tmp75, tmp78)
    tmp80 = tmp64 - tmp79
    tmp81 = -tmp80
    tmp82 = tmp61 + tmp81
    tmp83 = 4.0
    tmp84 = tmp82 / tmp83
    tmp89 = triton_helpers.maximum(tmp86, tmp88)
    tmp92 = triton_helpers.maximum(tmp89, tmp91)
    tmp95 = triton_helpers.maximum(tmp92, tmp94)
    tmp96 = tl_math.abs(tmp95)
    tmp97 = float("inf")
    tmp98 = tmp96 == tmp97
    tmp99 = 0.0
    tmp100 = tl.where(tmp98, tmp99, tmp95)
    tmp101 = tmp86 - tmp100
    tmp102 = tl_math.exp(tmp101)
    tmp103 = tmp88 - tmp100
    tmp104 = tl_math.exp(tmp103)
    tmp105 = tmp102 + tmp104
    tmp106 = tmp91 - tmp100
    tmp107 = tl_math.exp(tmp106)
    tmp108 = tmp105 + tmp107
    tmp109 = tmp94 - tmp100
    tmp110 = tl_math.exp(tmp109)
    tmp111 = tmp108 + tmp110
    tmp112 = tl_math.log(tmp111)
    tmp113 = tmp112 + tmp100
    tmp118 = triton_helpers.maximum(tmp115, tmp117)
    tmp121 = triton_helpers.maximum(tmp118, tmp120)
    tmp124 = triton_helpers.maximum(tmp121, tmp123)
    tmp125 = tl_math.abs(tmp124)
    tmp126 = tmp125 == tmp97
    tmp127 = tl.where(tmp126, tmp99, tmp124)
    tmp128 = tmp115 - tmp127
    tmp129 = tl_math.exp(tmp128)
    tmp130 = tmp117 - tmp127
    tmp131 = tl_math.exp(tmp130)
    tmp132 = tmp129 + tmp131
    tmp133 = tmp120 - tmp127
    tmp134 = tl_math.exp(tmp133)
    tmp135 = tmp132 + tmp134
    tmp136 = tmp123 - tmp127
    tmp137 = tl_math.exp(tmp136)
    tmp138 = tmp135 + tmp137
    tmp139 = tl_math.log(tmp138)
    tmp140 = tmp139 + tmp127
    tmp141 = tmp113 + tmp140
    tmp146 = triton_helpers.maximum(tmp143, tmp145)
    tmp149 = triton_helpers.maximum(tmp146, tmp148)
    tmp152 = triton_helpers.maximum(tmp149, tmp151)
    tmp153 = tl_math.abs(tmp152)
    tmp154 = tmp153 == tmp97
    tmp155 = tl.where(tmp154, tmp99, tmp152)
    tmp156 = tmp143 - tmp155
    tmp157 = tl_math.exp(tmp156)
    tmp158 = tmp145 - tmp155
    tmp159 = tl_math.exp(tmp158)
    tmp160 = tmp157 + tmp159
    tmp161 = tmp148 - tmp155
    tmp162 = tl_math.exp(tmp161)
    tmp163 = tmp160 + tmp162
    tmp164 = tmp151 - tmp155
    tmp165 = tl_math.exp(tmp164)
    tmp166 = tmp163 + tmp165
    tmp167 = tl_math.log(tmp166)
    tmp168 = tmp167 + tmp155
    tmp169 = tmp141 + tmp168
    tmp174 = triton_helpers.maximum(tmp171, tmp173)
    tmp177 = triton_helpers.maximum(tmp174, tmp176)
    tmp180 = triton_helpers.maximum(tmp177, tmp179)
    tmp181 = tl_math.abs(tmp180)
    tmp182 = tmp181 == tmp97
    tmp183 = tl.where(tmp182, tmp99, tmp180)
    tmp184 = tmp171 - tmp183
    tmp185 = tl_math.exp(tmp184)
    tmp186 = tmp173 - tmp183
    tmp187 = tl_math.exp(tmp186)
    tmp188 = tmp185 + tmp187
    tmp189 = tmp176 - tmp183
    tmp190 = tl_math.exp(tmp189)
    tmp191 = tmp188 + tmp190
    tmp192 = tmp179 - tmp183
    tmp193 = tl_math.exp(tmp192)
    tmp194 = tmp191 + tmp193
    tmp195 = tl_math.log(tmp194)
    tmp196 = tmp195 + tmp183
    tmp197 = tmp169 + tmp196
    tmp198 = tmp197 / tmp83
    tmp199 = tmp84 + tmp198
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp84, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp198, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp199, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    assert_size_stride(arg2_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (1, 4, 4), (16, 4, 1), 0), reinterpret_tensor(arg2_1, (1, 4, 4), (0, 1, 4), 0), out=buf0)
        buf1 = reinterpret_tensor(buf0, (4, 4), (4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [add_1, neg_dist, mul_1, sub_1], Original ATen: [aten.add, aten.div, aten.mul, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sub_0.run(buf1, arg0_1, arg2_1, 16, grid=grid(16), stream=stream0)
        del arg2_1
        buf2 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 1, 4), (4, 4, 1), 0), reinterpret_tensor(arg1_1, (4, 4, 1), (4, 1, 1), 0), out=buf2)
        buf3 = reinterpret_tensor(buf2, (4, ), (1, ), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [einsum, ref_sq, einsum_1, pos_sq, add, pos_dist, mul, sub], Original ATen: [aten.sum, aten.div, aten.add, aten.mul, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sub_sum_1.run(buf3, arg0_1, arg1_1, 4, grid=grid(4), stream=stream0)
        del arg0_1
        del arg1_1
        buf5 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [neg_dist_1, max_1, neg_dist_2], Original ATen: [aten.neg, aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_neg_sub_2.run(buf1, buf5, 16, grid=grid(16), stream=stream0)
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf7 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [pos_dist_1, pos_dist_2, neg_2, align, logsumexp, uniform, add_2], Original ATen: [aten.neg, aten.sub, aten.mean, aten.logsumexp, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_logsumexp_mean_neg_sub_3.run(buf3, buf1, buf5, buf4, buf6, buf7, 1, grid=grid(1), stream=stream0)
        del buf1
        del buf3
        del buf5
    return (buf7, buf4, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
