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


# kernel path: inductor_cache/yt/cytstzv6lfzr5d77mxh24lzxtskyr4i3awdbirvabogkvngt4q4f.py
# Topologically Sorted Source Nodes: [mean, norm, loss_act, setitem, mean_1, loss_bkg, add, pow_1, loss_um], Original ATen: [aten.mean, aten.linalg_vector_norm, aten.rsub, aten.lift_fresh, aten.index_put, aten.add, aten.pow]
# Source node to ATen node mapping:
#   add => add
#   loss_act => sub_4
#   loss_bkg => pow_3, pow_4, sum_4
#   loss_um => mean_4
#   mean => mean_2
#   mean_1 => mean_3
#   norm => pow_1, pow_2, sum_3
#   pow_1 => pow_5
#   setitem => full_default_5, index_put
# Graph fragment:
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%arg3_1, [1]), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mean_2, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (4, %pow_2), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%sub_4, [%lt], %full_default_5), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%arg4_1, [1]), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mean_3, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1]), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%index_put, %pow_4), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean_4 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%pow_5,), kwargs = {})
triton_per_fused_add_index_put_lift_fresh_linalg_vector_norm_mean_pow_rsub_0 = async_compile.triton('triton_per_fused_add_index_put_lift_fresh_linalg_vector_norm_mean_pow_rsub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_index_put_lift_fresh_linalg_vector_norm_mean_pow_rsub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_index_put_lift_fresh_linalg_vector_norm_mean_pow_rsub_0(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 4)
    r1 = rindex // 4
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr0 + (4 + r0 + 64*r1), None)
    tmp11 = tl.load(in_ptr0 + (20 + r0 + 64*r1), None)
    tmp13 = tl.load(in_ptr0 + (36 + r0 + 64*r1), None)
    tmp15 = tl.load(in_ptr0 + (52 + r0 + 64*r1), None)
    tmp20 = tl.load(in_ptr0 + (8 + r0 + 64*r1), None)
    tmp21 = tl.load(in_ptr0 + (24 + r0 + 64*r1), None)
    tmp23 = tl.load(in_ptr0 + (40 + r0 + 64*r1), None)
    tmp25 = tl.load(in_ptr0 + (56 + r0 + 64*r1), None)
    tmp30 = tl.load(in_ptr0 + (12 + r0 + 64*r1), None)
    tmp31 = tl.load(in_ptr0 + (28 + r0 + 64*r1), None)
    tmp33 = tl.load(in_ptr0 + (44 + r0 + 64*r1), None)
    tmp35 = tl.load(in_ptr0 + (60 + r0 + 64*r1), None)
    tmp45 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp46 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp48 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp50 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp54 = tl.load(in_ptr1 + (4 + r0 + 64*r1), None)
    tmp55 = tl.load(in_ptr1 + (20 + r0 + 64*r1), None)
    tmp57 = tl.load(in_ptr1 + (36 + r0 + 64*r1), None)
    tmp59 = tl.load(in_ptr1 + (52 + r0 + 64*r1), None)
    tmp64 = tl.load(in_ptr1 + (8 + r0 + 64*r1), None)
    tmp65 = tl.load(in_ptr1 + (24 + r0 + 64*r1), None)
    tmp67 = tl.load(in_ptr1 + (40 + r0 + 64*r1), None)
    tmp69 = tl.load(in_ptr1 + (56 + r0 + 64*r1), None)
    tmp74 = tl.load(in_ptr1 + (12 + r0 + 64*r1), None)
    tmp75 = tl.load(in_ptr1 + (28 + r0 + 64*r1), None)
    tmp77 = tl.load(in_ptr1 + (44 + r0 + 64*r1), None)
    tmp79 = tl.load(in_ptr1 + (60 + r0 + 64*r1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 * tmp8
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 / tmp7
    tmp18 = tmp17 * tmp17
    tmp19 = tmp9 + tmp18
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26 / tmp7
    tmp28 = tmp27 * tmp27
    tmp29 = tmp19 + tmp28
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36 / tmp7
    tmp38 = tmp37 * tmp37
    tmp39 = tmp29 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tmp7 - tmp40
    tmp42 = 0.0
    tmp43 = tmp41 < tmp42
    tmp44 = tl.where(tmp43, tmp42, tmp41)
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp51 = tmp49 + tmp50
    tmp52 = tmp51 / tmp7
    tmp53 = tmp52 * tmp52
    tmp56 = tmp54 + tmp55
    tmp58 = tmp56 + tmp57
    tmp60 = tmp58 + tmp59
    tmp61 = tmp60 / tmp7
    tmp62 = tmp61 * tmp61
    tmp63 = tmp53 + tmp62
    tmp66 = tmp64 + tmp65
    tmp68 = tmp66 + tmp67
    tmp70 = tmp68 + tmp69
    tmp71 = tmp70 / tmp7
    tmp72 = tmp71 * tmp71
    tmp73 = tmp63 + tmp72
    tmp76 = tmp74 + tmp75
    tmp78 = tmp76 + tmp77
    tmp80 = tmp78 + tmp79
    tmp81 = tmp80 / tmp7
    tmp82 = tmp81 * tmp81
    tmp83 = tmp73 + tmp82
    tmp84 = libdevice.sqrt(tmp83)
    tmp85 = tmp44 + tmp84
    tmp86 = tmp85 * tmp85
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK, RBLOCK])
    tmp89 = tl.sum(tmp87, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp89, None)
''', device_str='cuda')


# kernel path: inductor_cache/3j/c3j742osrn7cshks5yupdvloveraqfwgg7cwsin23lgeybv4hcoe.py
# Topologically Sorted Source Nodes: [sum_1, label, loss_cls, loss_bkg, add, pow_1, loss_um, mul, add_1, label_bkg, sum_2, label_bkg_1, loss_be, mul_1, loss_total], Original ATen: [aten.sum, aten.div, aten.binary_cross_entropy, aten.linalg_vector_norm, aten.add, aten.pow, aten.mean, aten.mul, aten.ones_like]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   label => div
#   label_bkg => full_default_2
#   label_bkg_1 => div_1
#   loss_be => full_default_3, full_default_4, log1p_1, log_1, maximum_2, maximum_3, mean_1, mul_2, mul_3, neg_1, sub_2, sub_3
#   loss_bkg => pow_4
#   loss_cls => full_default, full_default_1, log, log1p, maximum, maximum_1, mean, mul, mul_1, neg, sub, sub_1
#   loss_total => add_2
#   loss_um => mean_4
#   mul => mul_4
#   mul_1 => mul_5
#   pow_1 => pow_5
#   sum_1 => sum_1
#   sum_2 => sum_2
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%arg0_1, [1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %sum_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, 1), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%arg1_1,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%neg,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log1p, %full_default), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %maximum), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%arg1_1,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log, %full_default_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %maximum_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%sub_1,), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%index_put, %pow_4), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean_4 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%pow_5,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_4, 4), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mul_4), kwargs = {})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4, 4], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%full_default_2, [1], True), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%full_default_2, %sum_2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, 1), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%arg2_1,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%neg_1,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_2 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log1p_1, %full_default_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %maximum_2), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%arg2_1,), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_3 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log_1, %full_default_4), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %maximum_3), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%sub_3,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_1, 4), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_5), kwargs = {})
triton_per_fused_add_binary_cross_entropy_div_linalg_vector_norm_mean_mul_ones_like_pow_sum_1 = async_compile.triton('triton_per_fused_add_binary_cross_entropy_div_linalg_vector_norm_mean_mul_ones_like_pow_sum_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': (7,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_binary_cross_entropy_div_linalg_vector_norm_mean_mul_ones_like_pow_sum_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_binary_cross_entropy_div_linalg_vector_norm_mean_mul_ones_like_pow_sum_1(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
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
    r0 = rindex
    r1 = (rindex % 16)
    r3 = rindex // 64
    tmp4 = tl.load(in_ptr0 + (r0), None)
    tmp17 = tl.load(in_ptr1 + (r0), None)
    tmp18 = tl.load(in_ptr1 + (r1 + 64*r3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (16 + r1 + 64*r3), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (32 + r1 + 64*r3), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (48 + r1 + 64*r3), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (r0), None)
    tmp41 = tl.load(in_out_ptr1 + (0))
    tmp42 = tl.broadcast_to(tmp41, [1])
    tmp0 = 1.0
    tmp1 = 4.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2 - tmp0
    tmp5 = -tmp4
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = -100.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp3 * tmp8
    tmp10 = tl_math.log(tmp4)
    tmp11 = triton_helpers.maximum(tmp10, tmp7)
    tmp12 = tmp2 * tmp11
    tmp13 = tmp9 - tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp17 / tmp24
    tmp26 = tmp25 - tmp0
    tmp28 = -tmp27
    tmp29 = libdevice.log1p(tmp28)
    tmp30 = triton_helpers.maximum(tmp29, tmp7)
    tmp31 = tmp26 * tmp30
    tmp32 = tl_math.log(tmp27)
    tmp33 = triton_helpers.maximum(tmp32, tmp7)
    tmp34 = tmp25 * tmp33
    tmp35 = tmp31 - tmp34
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp39 = 256.0
    tmp40 = tmp38 / tmp39
    tmp43 = 16.0
    tmp44 = tmp42 / tmp43
    tmp45 = tmp16 / tmp39
    tmp46 = tmp44 * tmp1
    tmp47 = tmp40 + tmp46
    tmp48 = tmp45 * tmp1
    tmp49 = tmp47 + tmp48
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp40, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([1], 0, tl.int32)), tmp44, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (tl.full([1], 0, tl.int32)), tmp45, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp49, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg3_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg4_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf6 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [mean, norm, loss_act, setitem, mean_1, loss_bkg, add, pow_1, loss_um], Original ATen: [aten.mean, aten.linalg_vector_norm, aten.rsub, aten.lift_fresh, aten.index_put, aten.add, aten.pow]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_index_put_lift_fresh_linalg_vector_norm_mean_pow_rsub_0.run(arg3_1, arg4_1, buf6, 1, 16, grid=grid(1), stream=stream0)
        del arg3_1
        del arg4_1
        buf8 = empty_strided_cuda((), (), torch.float32)
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1; del buf1  # reuse
        buf7 = buf6; del buf6  # reuse
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sum_1, label, loss_cls, loss_bkg, add, pow_1, loss_um, mul, add_1, label_bkg, sum_2, label_bkg_1, loss_be, mul_1, loss_total], Original ATen: [aten.sum, aten.div, aten.binary_cross_entropy, aten.linalg_vector_norm, aten.add, aten.pow, aten.mean, aten.mul, aten.ones_like]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_binary_cross_entropy_div_linalg_vector_norm_mean_mul_ones_like_pow_sum_1.run(buf2, buf7, buf9, arg2_1, arg0_1, arg1_1, buf10, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf10, buf2, buf9, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
