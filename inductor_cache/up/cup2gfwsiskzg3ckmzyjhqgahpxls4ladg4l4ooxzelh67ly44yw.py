# AOT ID: ['46_forward']
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


# kernel path: inductor_cache/2r/c2rbkqrb4kvvv2ve5bsw2b3dlosx463pjt4eiwguv3k5xvkkpfgb.py
# Topologically Sorted Source Nodes: [cdist], Original ATen: [aten._euclidean_dist]
# Source node to ATen node mapping:
#   cdist => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mul, %sum_1, %full_default], -1), kwargs = {})
triton_poi_fused__euclidean_dist_0 = async_compile.triton('triton_poi_fused__euclidean_dist_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__euclidean_dist_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__euclidean_dist_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = xindex // 6
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16*(x0) + 64*(x1 // 16) + ((x1 % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = -2.0
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 5, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr0 + (64*(x1 // 16) + ((x1 % 16))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 * tmp14
    tmp16 = tl.load(in_ptr0 + (16 + 64*(x1 // 16) + ((x1 % 16))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = tl.load(in_ptr0 + (32 + 64*(x1 // 16) + ((x1 % 16))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tl.load(in_ptr0 + (48 + 64*(x1 // 16) + ((x1 % 16))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp13, tmp24, tmp25)
    tmp27 = tmp0 >= tmp11
    tmp28 = tl.full([1], 6, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = 1.0
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tl.where(tmp13, tmp26, tmp32)
    tmp34 = tl.where(tmp4, tmp9, tmp33)
    tl.store(out_ptr0 + (x2), tmp34, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uk/cukzm3ty466tsx6tvim2vhy5du5pvmaxdrtreh7vl3fjuaogmnfw.py
# Topologically Sorted Source Nodes: [cdist], Original ATen: [aten._euclidean_dist]
# Source node to ATen node mapping:
#   cdist => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_2, %full_default_1, %sum_2], -1), kwargs = {})
triton_poi_fused__euclidean_dist_1 = async_compile.triton('triton_poi_fused__euclidean_dist_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__euclidean_dist_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__euclidean_dist_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = xindex // 6
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 5, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = 1.0
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp0 >= tmp7
    tmp14 = tl.full([1], 6, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr0 + (4*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 * tmp16
    tmp18 = tl.load(in_ptr0 + (1 + 4*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 + tmp19
    tmp21 = tl.load(in_ptr0 + (2 + 4*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 + tmp22
    tmp24 = tl.load(in_ptr0 + (3 + 4*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp13, tmp26, tmp27)
    tmp29 = tl.where(tmp9, tmp12, tmp28)
    tmp30 = tl.where(tmp4, tmp5, tmp29)
    tl.store(out_ptr0 + (x2), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g7/cg7crnlor3u23dwyualwy5tgz5jh46p5hlb7gajkoqplsxu3o2el.py
# Topologically Sorted Source Nodes: [cdist, min_encoding_indices], Original ATen: [aten._euclidean_dist, aten.argmin]
# Source node to ATen node mapping:
#   cdist => clamp_min, sqrt
#   min_encoding_indices => argmin
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mm, 0), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
#   %argmin : [num_users=2] = call_function[target=torch.ops.aten.argmin.default](args = (%sqrt, 1), kwargs = {})
triton_poi_fused__euclidean_dist_argmin_2 = async_compile.triton('triton_poi_fused__euclidean_dist_argmin_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__euclidean_dist_argmin_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__euclidean_dist_argmin_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = libdevice.sqrt(tmp2)
    tmp5 = triton_helpers.maximum(tmp4, tmp1)
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp3 == tmp6
    tmp9 = tmp3 != tmp3
    tmp10 = tmp6 != tmp6
    tmp11 = tmp9 > tmp10
    tmp12 = tmp7 | tmp11
    tmp13 = tmp9 & tmp10
    tmp14 = tmp8 | tmp13
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp14 & tmp17
    tmp19 = tmp12 | tmp18
    tmp20 = tl.where(tmp19, tmp3, tmp6)
    tmp21 = tl.where(tmp19, tmp15, tmp16)
    tmp23 = triton_helpers.maximum(tmp22, tmp1)
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tmp20 < tmp24
    tmp26 = tmp20 == tmp24
    tmp27 = tmp20 != tmp20
    tmp28 = tmp24 != tmp24
    tmp29 = tmp27 > tmp28
    tmp30 = tmp25 | tmp29
    tmp31 = tmp27 & tmp28
    tmp32 = tmp26 | tmp31
    tmp33 = tl.full([1], 2, tl.int64)
    tmp34 = tmp21 < tmp33
    tmp35 = tmp32 & tmp34
    tmp36 = tmp30 | tmp35
    tmp37 = tl.where(tmp36, tmp20, tmp24)
    tmp38 = tl.where(tmp36, tmp21, tmp33)
    tmp40 = triton_helpers.maximum(tmp39, tmp1)
    tmp41 = libdevice.sqrt(tmp40)
    tmp42 = tmp37 < tmp41
    tmp43 = tmp37 == tmp41
    tmp44 = tmp37 != tmp37
    tmp45 = tmp41 != tmp41
    tmp46 = tmp44 > tmp45
    tmp47 = tmp42 | tmp46
    tmp48 = tmp44 & tmp45
    tmp49 = tmp43 | tmp48
    tmp50 = tl.full([1], 3, tl.int64)
    tmp51 = tmp38 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tmp47 | tmp52
    tmp54 = tl.where(tmp53, tmp37, tmp41)
    tmp55 = tl.where(tmp53, tmp38, tmp50)
    tl.store(out_ptr0 + (x0), tmp55, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s5/cs5up6j5ahbm2jdprfcjxlwh5lrppcxtgo5ioycgxk44vzq43hte.py
# Topologically Sorted Source Nodes: [z, sub, pow_1, mean, mul, loss, z_q_2], Original ATen: [aten.clone, aten.sub, aten.pow, aten.mean, aten.mul, aten.add]
# Source node to ATen node mapping:
#   loss => add
#   mean => mean
#   mul => mul_1
#   pow_1 => pow_3
#   sub => sub
#   z => clone
#   z_q_2 => clone_1
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %clone), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 4), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mul_1), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused_add_clone_mean_mul_pow_sub_3 = async_compile.triton('triton_per_fused_add_clone_mean_mul_pow_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_mean_mul_pow_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clone_mean_mul_pow_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r2 = rindex // 64
    r1 = ((rindex // 16) % 4)
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 16*r2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r3), None)
    tmp1 = tl.full([RBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 4), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (r1 + 4*tmp4), None, eviction_policy='evict_last')
    tmp8 = tmp6 - tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp7 + tmp8
    tmp14 = 256.0
    tmp15 = tmp12 / tmp14
    tmp16 = 4.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp15 + tmp17
    tl.store(out_ptr0 + (tl.broadcast_to(r3, [RBLOCK])), tmp13, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/wr/cwr6wkbzak7jyvcodbfs7auiyyrcs3odgn6snuxwmszk3svk35oy.py
# Topologically Sorted Source Nodes: [z, sub], Original ATen: [aten.clone, aten.sub, aten.pow, aten.mul]
# Source node to ATen node mapping:
#   sub => sub
#   z => clone
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %clone), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 1.0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_5, 2.0), kwargs = {})
triton_poi_fused_clone_mul_pow_sub_4 = async_compile.triton('triton_poi_fused_clone_mul_pow_sub_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_pow_sub_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_pow_sub_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y3 = yindex
    x2 = xindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + 16*x2 + 64*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(ymask), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (x2 + 4*tmp4), xmask & ymask)
    tmp8 = tmp6 - tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + (x2 + 4*y3), tmp10, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 6), (6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cdist], Original ATen: [aten._euclidean_dist]
        stream0 = get_raw_stream(0)
        triton_poi_fused__euclidean_dist_0.run(primals_1, buf0, 384, grid=grid(384), stream=stream0)
        buf1 = empty_strided_cuda((4, 6), (6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cdist], Original ATen: [aten._euclidean_dist]
        stream0 = get_raw_stream(0)
        triton_poi_fused__euclidean_dist_1.run(primals_2, buf1, 24, grid=grid(24), stream=stream0)
        buf2 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cdist], Original ATen: [aten._euclidean_dist]
        extern_kernels.mm(buf0, reinterpret_tensor(buf1, (6, 4), (1, 6), 0), out=buf2)
        del buf0
        del buf1
        buf3 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [cdist, min_encoding_indices], Original ATen: [aten._euclidean_dist, aten.argmin]
        stream0 = get_raw_stream(0)
        triton_poi_fused__euclidean_dist_argmin_2.run(buf2, buf3, 64, grid=grid(64), stream=stream0)
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = reinterpret_tensor(buf2, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf2  # reuse
        buf7 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [z, sub, pow_1, mean, mul, loss, z_q_2], Original ATen: [aten.clone, aten.sub, aten.pow, aten.mean, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_mean_mul_pow_sub_3.run(buf7, buf3, primals_2, primals_1, buf5, 1, 256, grid=grid(1), stream=stream0)
        buf6 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z, sub], Original ATen: [aten.clone, aten.sub, aten.pow, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_pow_sub_4.run(buf3, primals_2, primals_1, buf6, 64, 4, grid=grid(64, 4), stream=stream0)
        del primals_1
        del primals_2
    return (buf5, buf7, buf3, buf3, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
