# AOT ID: ['45_inference']
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


# kernel path: inductor_cache/ku/ckuxiazspzhaucg652pxgspjqpdxbujvpcutsozof3f42rqopyr4.py
# Topologically Sorted Source Nodes: [feature_matrix, feature_matrix_1], Original ATen: [aten.cat, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   feature_matrix => cat
#   feature_matrix_1 => pow_1, sum_1
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%arg0_1, %arg1_1],), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%cat, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
triton_poi_fused_cat_linalg_vector_norm_0 = async_compile.triton('triton_poi_fused_cat_linalg_vector_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_linalg_vector_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_linalg_vector_norm_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x0 = (xindex % 16)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1)), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-4) + x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10 * tmp10
    tmp12 = tl.load(in_ptr0 + (16 + x0 + 64*(x1)), tmp4 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr1 + (16 + x0 + 64*((-4) + x1)), tmp6 & xmask, other=0.0)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp14 * tmp14
    tmp16 = tmp11 + tmp15
    tmp17 = tl.load(in_ptr0 + (32 + x0 + 64*(x1)), tmp4 & xmask, other=0.0)
    tmp18 = tl.load(in_ptr1 + (32 + x0 + 64*((-4) + x1)), tmp6 & xmask, other=0.0)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp19 * tmp19
    tmp21 = tmp16 + tmp20
    tmp22 = tl.load(in_ptr0 + (48 + x0 + 64*(x1)), tmp4 & xmask, other=0.0)
    tmp23 = tl.load(in_ptr1 + (48 + x0 + 64*((-4) + x1)), tmp6 & xmask, other=0.0)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp24 * tmp24
    tmp26 = tmp21 + tmp25
    tl.store(out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/43/c43yqixq2bqhkp3objfegoswfl7hdkcp3z6ewfa2rwxljve33zkn.py
# Topologically Sorted Source Nodes: [feature_matrix, feature_matrix_1], Original ATen: [aten.cat, aten.div]
# Source node to ATen node mapping:
#   feature_matrix => cat
#   feature_matrix_1 => div
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%arg0_1, %arg1_1],), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%cat, %expand), kwargs = {})
triton_poi_fused_cat_div_1 = async_compile.triton('triton_poi_fused_cat_div_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_div_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_div_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 64
    x3 = (xindex % 64)
    x0 = (xindex % 16)
    x4 = xindex
    tmp11 = tl.load(in_ptr2 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + 64*(x2)), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x3 + 64*((-4) + x2)), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-12
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/if/cifq5roriyhvlz265uvrlia6nshf2nicnle6bspmxo2alk7loqxo.py
# Topologically Sorted Source Nodes: [cosine_similarity], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul]
# Source node to ATen node mapping:
#   cosine_similarity => clamp_min_1, clamp_min_2, div_3, div_4, mul, pow_4, pow_5, pow_6, pow_7, sum_3, sum_4
# Graph fragment:
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_4, [1], True), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_5, 1e-08), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %clamp_min_1), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_6, [1], True), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_7, 1e-08), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg1_1, %clamp_min_2), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %div_3), kwargs = {})
triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_2 = async_compile.triton('triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (x3), xmask)
    tmp17 = tl.load(in_ptr1 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr1 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-08
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tmp18 = tmp17 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = triton_helpers.maximum(tmp28, tmp13)
    tmp30 = tmp16 / tmp29
    tmp31 = tmp15 * tmp30
    tl.store(out_ptr0 + (x3), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ly/clynjrsxty36pfbzbhpplg4ze4s2qs3dwvvedpyupnk3zuid3r7e.py
# Topologically Sorted Source Nodes: [neg, pow_1, sub, cosine_matrix, truediv_1, exp_cosine_matrix, sum_1, exp_sim_sum, neg_loss, loss, loss_1], Original ATen: [aten.neg, aten.pow, aten.rsub, aten.div, aten.exp, aten.sum, aten.sub, aten.log, aten.add, aten.mean]
# Source node to ATen node mapping:
#   cosine_matrix => div_1
#   exp_cosine_matrix => exp
#   exp_sim_sum => sub_1
#   loss => add
#   loss_1 => mean
#   neg => neg
#   neg_loss => log
#   pow_1 => pow_3
#   sub => sub
#   sum_1 => sum_2
#   truediv_1 => div_2
# Graph fragment:
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%view,), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%_cdist_forward, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (2, %pow_3), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, 2), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_1, 4), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_2,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1]), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_2, 1.2840254166877414), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sub_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %log), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add,), kwargs = {})
triton_per_fused_add_div_exp_log_mean_neg_pow_rsub_sub_sum_3 = async_compile.triton('triton_per_fused_add_div_exp_log_mean_neg_pow_rsub_sub_sum_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_exp_log_mean_neg_pow_rsub_sub_sum_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_exp_log_mean_neg_pow_rsub_sub_sum_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp16 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp23 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp32 = tl.load(in_ptr1 + (r0 + 64*((r1 % 4))), None)
    tmp33 = tl.load(in_ptr1 + (16 + r0 + 64*((r1 % 4))), None)
    tmp35 = tl.load(in_ptr1 + (32 + r0 + 64*((r1 % 4))), None)
    tmp37 = tl.load(in_ptr1 + (48 + r0 + 64*((r1 % 4))), None)
    tmp1 = tmp0 * tmp0
    tmp2 = 2.0
    tmp3 = tmp2 - tmp1
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = 0.25
    tmp7 = tmp5 * tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp10 = tmp9 * tmp9
    tmp11 = tmp2 - tmp10
    tmp12 = tmp11 * tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp8 + tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = tmp2 - tmp17
    tmp19 = tmp18 * tmp4
    tmp20 = tmp19 * tmp6
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp15 + tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp2 - tmp24
    tmp26 = tmp25 * tmp4
    tmp27 = tmp26 * tmp6
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp22 + tmp28
    tmp30 = 1.2840254166877414
    tmp31 = tmp29 - tmp30
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38 * tmp6
    tmp40 = -tmp39
    tmp41 = tl_math.log(tmp31)
    tmp42 = tmp40 + tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.sum(tmp43, 1)[:, None]
    tmp46 = 128.0
    tmp47 = tmp45 / tmp46
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp47, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((8, 1, 4, 4), (16, 128, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [feature_matrix, feature_matrix_1], Original ATen: [aten.cat, aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_linalg_vector_norm_0.run(arg0_1, arg1_1, buf1, 128, grid=grid(128), stream=stream0)
        buf2 = empty_strided_cuda((8, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [feature_matrix, feature_matrix_1], Original ATen: [aten.cat, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_div_1.run(arg0_1, arg1_1, buf1, buf2, 512, grid=grid(512), stream=stream0)
        del buf1
        # Topologically Sorted Source Nodes: [feature_matrix, feature_matrix_1, cdist], Original ATen: [aten.cat, aten.div, aten._cdist_forward]
        buf3 = torch.ops.aten._cdist_forward.default(buf2, buf2, 2.0, None)
        del buf2
        buf4 = buf3
        del buf3
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cosine_similarity], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_2.run(arg0_1, arg1_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del arg1_1
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [neg, pow_1, sub, cosine_matrix, truediv_1, exp_cosine_matrix, sum_1, exp_sim_sum, neg_loss, loss, loss_1], Original ATen: [aten.neg, aten.pow, aten.rsub, aten.div, aten.exp, aten.sum, aten.sub, aten.log, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_exp_log_mean_neg_pow_rsub_sub_sum_3.run(buf7, buf4, buf0, 1, 128, grid=grid(1), stream=stream0)
        del buf0
        del buf4
    return (buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
