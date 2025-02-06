# AOT ID: ['0_inference']
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


# kernel path: inductor_cache/f3/cf3mb6avrtkisakf3b5uietddb6xsyikuvtbwnqidgb7fmcpozk2.py
# Topologically Sorted Source Nodes: [max_1, input_1, input_2, sort, pow_1, cumsum_1, cumsum], Original ATen: [aten.max, aten.sub, aten.div, aten.sort, aten.pow, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum => cumsum
#   cumsum_1 => cumsum_1
#   input_1 => sub
#   input_2 => div
#   max_1 => max_1
#   pow_1 => pow_1
#   sort => sort
# Graph fragment:
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%arg0_1, -1, True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %getitem), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, 2), kwargs = {})
#   %sort : [num_users=1] = call_function[target=torch.ops.aten.sort.default](args = (%div, -1, True), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%getitem_2, 2), kwargs = {})
#   %cumsum_1 : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%pow_1, -1), kwargs = {})
#   %cumsum : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%getitem_2, -1), kwargs = {})
triton_per_fused_cumsum_div_max_pow_sort_sub_0 = async_compile.triton('triton_per_fused_cumsum_div_max_pow_sort_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_div_max_pow_sort_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cumsum_div_max_pow_sort_sub_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = r1
    tmp12 = tmp11.to(tl.int16)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp14 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15, tmp16, = triton_helpers.sort_with_index(tmp13, tmp14, None, 1, stable=False, descending=True)
    tmp17 = tmp15 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp20, = tl.associative_scan((tmp19,), 1, _triton_helper_fn_add0)
    tmp21 = tmp15.to(tl.float32)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp23, = tl.associative_scan((tmp22,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r1 + 4*x0), tmp10, xmask)
    tl.store(out_ptr1 + (r1 + 4*x0), tmp15, xmask)
    tl.store(out_ptr2 + (r1 + 4*x0), tmp20, xmask)
    tl.store(out_ptr3 + (r1 + 4*x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/db/cdbwj5mhvhafkoikr3jwc4h3q6brt4xfib3q33ti3hmdwhaf7xgd.py
# Topologically Sorted Source Nodes: [mean_sq, mean, pow_2, sub_1, ss, sub_2, delta, delta_nz, sqrt, tau, le, sum_1], Original ATen: [aten.div, aten.pow, aten.sub, aten.mul, aten.rsub, aten.clamp, aten.sqrt, aten.le, aten.sum]
# Source node to ATen node mapping:
#   delta => div_3
#   delta_nz => clamp_min
#   le => le
#   mean => div_1
#   mean_sq => div_2
#   pow_2 => pow_2
#   sqrt => sqrt
#   ss => mul_1
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sum_1 => sum_1
#   tau => sub_3
# Graph fragment:
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%cumsum_1, %permute), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%cumsum, %permute), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_1, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_2, %pow_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute, %sub_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_1), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, %permute), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_3, 0), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %sqrt), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%sub_3, %getitem_2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%le, [-1]), kwargs = {})
triton_poi_fused_clamp_div_le_mul_pow_rsub_sqrt_sub_sum_1 = async_compile.triton('triton_poi_fused_clamp_div_le_mul_pow_rsub_sqrt_sub_sum_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_le_mul_pow_rsub_sqrt_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_le_mul_pow_rsub_sqrt_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp3 / tmp1
    tmp5 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp1 - tmp7
    tmp9 = tmp8 / tmp1
    tmp10 = 0.0
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tmp2 - tmp12
    tmp15 = tmp13 <= tmp14
    tmp16 = tmp15.to(tl.int64)
    tmp18 = 2.0
    tmp19 = tmp17 / tmp18
    tmp21 = tmp20 / tmp18
    tmp22 = tmp19 * tmp19
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tmp1 - tmp24
    tmp26 = tmp25 / tmp18
    tmp27 = triton_helpers.maximum(tmp26, tmp10)
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tmp19 - tmp28
    tmp31 = tmp29 <= tmp30
    tmp32 = tmp31.to(tl.int64)
    tmp33 = tmp16 + tmp32
    tmp35 = 3.0
    tmp36 = tmp34 / tmp35
    tmp38 = tmp37 / tmp35
    tmp39 = tmp36 * tmp36
    tmp40 = tmp38 - tmp39
    tmp41 = tmp35 * tmp40
    tmp42 = tmp1 - tmp41
    tmp43 = tmp42 / tmp35
    tmp44 = triton_helpers.maximum(tmp43, tmp10)
    tmp45 = libdevice.sqrt(tmp44)
    tmp46 = tmp36 - tmp45
    tmp48 = tmp46 <= tmp47
    tmp49 = tmp48.to(tl.int64)
    tmp50 = tmp33 + tmp49
    tmp52 = 4.0
    tmp53 = tmp51 / tmp52
    tmp55 = tmp54 / tmp52
    tmp56 = tmp53 * tmp53
    tmp57 = tmp55 - tmp56
    tmp58 = tmp52 * tmp57
    tmp59 = tmp1 - tmp58
    tmp60 = tmp59 / tmp52
    tmp61 = triton_helpers.maximum(tmp60, tmp10)
    tmp62 = libdevice.sqrt(tmp61)
    tmp63 = tmp53 - tmp62
    tmp65 = tmp63 <= tmp64
    tmp66 = tmp65.to(tl.int64)
    tmp67 = tmp50 + tmp66
    tl.store(out_ptr0 + (x0), tmp67, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wc/cwcpw4vaqf3p7lx25j2oxazhppscevwp4dibb7xmt5jipjzsardm.py
# Topologically Sorted Source Nodes: [mean_sq, mean, pow_2, sub_1, ss, sub_2, delta, delta_nz, sqrt, tau, sub_4, tau_star, sub_5, clamp_1, output], Original ATen: [aten.div, aten.pow, aten.sub, aten.mul, aten.rsub, aten.clamp, aten.sqrt, aten.gather]
# Source node to ATen node mapping:
#   clamp_1 => clamp_min_1
#   delta => div_3
#   delta_nz => clamp_min
#   mean => div_1
#   mean_sq => div_2
#   output => pow_3
#   pow_2 => pow_2
#   sqrt => sqrt
#   ss => mul_1
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_4 => sub_4
#   sub_5 => sub_5
#   tau => sub_3
#   tau_star => gather
# Graph fragment:
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%cumsum_1, %permute), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%cumsum, %permute), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_1, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_2, %pow_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute, %sub_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_1), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, %permute), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_3, 0), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %sqrt), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, 1), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_3, -1, %sub_4), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %gather), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_1, 2), kwargs = {})
triton_poi_fused_clamp_div_gather_mul_pow_rsub_sqrt_sub_2 = async_compile.triton('triton_poi_fused_clamp_div_gather_mul_pow_rsub_sqrt_sub_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_gather_mul_pow_rsub_sqrt_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_gather_mul_pow_rsub_sqrt_sub_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([XBLOCK], 4, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 4")
    tmp9 = tl.load(in_ptr1 + (tmp7 + 4*x1), xmask, eviction_policy='evict_last')
    tmp10 = 1 + tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tl.load(in_ptr2 + (tmp7 + 4*x1), xmask, eviction_policy='evict_last')
    tmp14 = tmp13 / tmp11
    tmp15 = tmp12 * tmp12
    tmp16 = tmp14 - tmp15
    tmp17 = tmp11 * tmp16
    tmp18 = 1.0
    tmp19 = tmp18 - tmp17
    tmp20 = tmp19 / tmp11
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp12 - tmp23
    tmp25 = tmp0 - tmp24
    tmp26 = triton_helpers.maximum(tmp25, tmp21)
    tmp27 = tmp26 * tmp26
    tl.store(in_out_ptr0 + (x2), tmp27, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_1, input_1, input_2, sort, pow_1, cumsum_1, cumsum], Original ATen: [aten.max, aten.sub, aten.div, aten.sort, aten.pow, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_div_max_pow_sort_sub_0.run(arg0_1, buf0, buf1, buf3, buf4, 64, 4, grid=grid(64), stream=stream0)
        del arg0_1
        buf5 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [mean_sq, mean, pow_2, sub_1, ss, sub_2, delta, delta_nz, sqrt, tau, le, sum_1], Original ATen: [aten.div, aten.pow, aten.sub, aten.mul, aten.rsub, aten.clamp, aten.sqrt, aten.le, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_div_le_mul_pow_rsub_sqrt_sub_sum_1.run(buf4, buf3, buf1, buf5, 64, grid=grid(64), stream=stream0)
        del buf1
        buf6 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mean_sq, mean, pow_2, sub_1, ss, sub_2, delta, delta_nz, sqrt, tau, sub_4, tau_star, sub_5, clamp_1, output], Original ATen: [aten.div, aten.pow, aten.sub, aten.mul, aten.rsub, aten.clamp, aten.sqrt, aten.gather]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_div_gather_mul_pow_rsub_sqrt_sub_2.run(buf6, buf5, buf4, buf3, 256, grid=grid(256), stream=stream0)
        del buf3
        del buf4
        del buf5
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
