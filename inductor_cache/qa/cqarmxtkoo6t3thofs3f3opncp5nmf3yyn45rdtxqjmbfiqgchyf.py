# AOT ID: ['8_inference']
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


# kernel path: inductor_cache/os/cosi24s25vgkvo4jrnfnny3pyowv7fciqidrjxuvkiohxzyn6hml.py
# Topologically Sorted Source Nodes: [distmat, distmat_1, distmat_2, addmm__2, mul_6, exp_6, K_6, mul_7, exp_7, K_7, mul_8, exp_8, K_8, mean_2], Original ATen: [aten.add, aten.addmm, aten.mul, aten.exp, aten.mean]
# Source node to ATen node mapping:
#   K_6 => add_9
#   K_7 => add_10
#   K_8 => add_11
#   addmm__2 => add_tensor
#   distmat => add
#   distmat_1 => add_4
#   distmat_2 => add_8
#   exp_6 => exp_6
#   exp_7 => exp_7
#   exp_8 => exp_8
#   mean_2 => mean_2
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand, %permute), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_2, %permute_2), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_4, %permute_4), kwargs = {})
#   %add_tensor : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %add_8), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, -0.5), kwargs = {})
#   %exp_6 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_6,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_6, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, -0.02), kwargs = {})
#   %exp_7 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_7,), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %exp_7), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, -0.005), kwargs = {})
#   %exp_8 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_8,), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %exp_8), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_11,), kwargs = {})
triton_per_fused_add_addmm_exp_mean_mul_0 = async_compile.triton('triton_per_fused_add_addmm_exp_mean_mul_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_exp_mean_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_addmm_exp_mean_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex // 4
    r0 = (rindex % 4)
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (4*r1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*r1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*r1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (4*r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + (1 + 4*r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (2 + 4*r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (3 + 4*r1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_out_ptr0 + (r2), None)
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
    tmp24 = tmp23 * tmp23
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 + tmp29
    tmp32 = tmp31 * tmp31
    tmp33 = tmp30 + tmp32
    tmp35 = tmp34 * tmp34
    tmp37 = tmp36 * tmp36
    tmp38 = tmp35 + tmp37
    tmp40 = tmp39 * tmp39
    tmp41 = tmp38 + tmp40
    tmp43 = tmp42 * tmp42
    tmp44 = tmp41 + tmp43
    tmp45 = tmp33 + tmp44
    tmp47 = tmp10 + tmp44
    tmp48 = tmp46 + tmp47
    tmp49 = -0.5
    tmp50 = tmp48 * tmp49
    tmp51 = tl_math.exp(tmp50)
    tmp52 = 0.0
    tmp53 = tmp51 + tmp52
    tmp54 = -0.02
    tmp55 = tmp48 * tmp54
    tmp56 = tl_math.exp(tmp55)
    tmp57 = tmp53 + tmp56
    tmp58 = -0.005
    tmp59 = tmp48 * tmp58
    tmp60 = tl_math.exp(tmp59)
    tmp61 = tmp57 + tmp60
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK, RBLOCK])
    tmp64 = tl.sum(tmp62, 1)[:, None]
    tl.store(out_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp22, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp45, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp64, None)
''', device_str='cuda')


# kernel path: inductor_cache/5f/c5frlkre2od4th5r447dzpxlychbmsdo6xdrymzttksck3y6kro7.py
# Topologically Sorted Source Nodes: [d_xx, mul, exp, K, mul_1, exp_1, K_1, mul_2, exp_2, K_2, mean, d_yy, mul_3, exp_3, K_3, mul_4, exp_4, K_4, mul_5, exp_5, K_5, mean_1, add_6, mul_6, exp_6, K_6, mul_7, exp_7, K_7, mul_8, exp_8, K_8, mean_2, mul_9, sub], Original ATen: [aten.stack, aten.mul, aten.exp, aten.add, aten.mean, aten.sub]
# Source node to ATen node mapping:
#   K => add_1
#   K_1 => add_2
#   K_2 => add_3
#   K_3 => add_5
#   K_4 => add_6
#   K_5 => add_7
#   K_6 => add_9
#   K_7 => add_10
#   K_8 => add_11
#   add_6 => add_12
#   d_xx => cat_2
#   d_yy => cat_5
#   exp => exp
#   exp_1 => exp_1
#   exp_2 => exp_2
#   exp_3 => exp_3
#   exp_4 => exp_4
#   exp_5 => exp_5
#   exp_6 => exp_6
#   exp_7 => exp_7
#   exp_8 => exp_8
#   mean => mean
#   mean_1 => mean_1
#   mean_2 => mean_2
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   sub => sub
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_4, %cat, %cat_1, %slice_15],), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, -0.5), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, -0.02), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_1,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %exp_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, -0.005), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_2,), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %exp_2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_3,), kwargs = {})
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_20, %cat_3, %cat_4, %slice_31],), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, -0.5), kwargs = {})
#   %exp_3 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_3,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_3, 0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, -0.02), kwargs = {})
#   %exp_4 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_4,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %exp_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, -0.005), kwargs = {})
#   %exp_5 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_5,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %exp_5), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_7,), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, -0.5), kwargs = {})
#   %exp_6 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_6,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_6, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, -0.02), kwargs = {})
#   %exp_7 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_7,), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %exp_7), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, -0.005), kwargs = {})
#   %exp_8 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_8,), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %exp_8), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_11,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_2, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %mul_9), kwargs = {})
triton_per_fused_add_exp_mean_mul_stack_sub_1 = async_compile.triton('triton_per_fused_add_exp_mean_mul_stack_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_exp_mean_mul_stack_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_exp_mean_mul_stack_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp16 = tl.load(in_ptr0 + (4))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp41 = tl.load(in_ptr0 + (11))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, RBLOCK])
    tmp71 = tl.load(in_ptr1 + (4))
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK, RBLOCK])
    tmp78 = tl.load(in_ptr1 + (11))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK, RBLOCK])
    tmp104 = tl.load(in_ptr2 + (0))
    tmp105 = tl.broadcast_to(tmp104, [XBLOCK, 1])
    tmp0 = r0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 3, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(1 + (r0), [XBLOCK, RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 6, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.broadcast_to((-3) + r0, [XBLOCK, RBLOCK])
    tmp11 = tl.full([1, 1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1, 1], 1, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp14 & tmp9
    tmp18 = tmp10 >= tmp13
    tmp19 = tl.full([1, 1], 3, tl.int64)
    tmp20 = tmp10 < tmp19
    tmp21 = tmp18 & tmp9
    tmp22 = tl.load(in_ptr0 + (tl.broadcast_to(6 + ((-1) + ((-3) + r0)), [XBLOCK, RBLOCK])), rmask & tmp21, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp14, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp9, tmp23, tmp24)
    tmp26 = tmp0 >= tmp7
    tmp27 = tl.full([1, 1], 9, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.broadcast_to((-6) + r0, [XBLOCK, RBLOCK])
    tmp31 = tl.full([1, 1], 0, tl.int64)
    tmp32 = tmp30 >= tmp31
    tmp33 = tl.full([1, 1], 2, tl.int64)
    tmp34 = tmp30 < tmp33
    tmp35 = tmp34 & tmp29
    tmp36 = tl.load(in_ptr0 + (tl.broadcast_to(8 + ((-6) + r0), [XBLOCK, RBLOCK])), rmask & tmp35, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp30 >= tmp33
    tmp38 = tl.full([1, 1], 3, tl.int64)
    tmp39 = tmp30 < tmp38
    tmp40 = tmp37 & tmp29
    tmp43 = tl.where(tmp34, tmp36, tmp42)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp29, tmp43, tmp44)
    tmp46 = tmp0 >= tmp27
    tmp47 = tl.full([1, 1], 12, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tl.load(in_ptr0 + (tl.broadcast_to(12 + ((-9) + r0), [XBLOCK, RBLOCK])), rmask & tmp46, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.where(tmp29, tmp45, tmp49)
    tmp51 = tl.where(tmp9, tmp25, tmp50)
    tmp52 = tl.where(tmp4, tmp5, tmp51)
    tmp53 = -0.5
    tmp54 = tmp52 * tmp53
    tmp55 = tl_math.exp(tmp54)
    tmp56 = 0.0
    tmp57 = tmp55 + tmp56
    tmp58 = -0.02
    tmp59 = tmp52 * tmp58
    tmp60 = tl_math.exp(tmp59)
    tmp61 = tmp57 + tmp60
    tmp62 = -0.005
    tmp63 = tmp52 * tmp62
    tmp64 = tl_math.exp(tmp63)
    tmp65 = tmp61 + tmp64
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, RBLOCK])
    tmp68 = tl.where(rmask, tmp66, 0)
    tmp69 = tl.sum(tmp68, 1)[:, None]
    tmp70 = tl.load(in_ptr1 + (tl.broadcast_to(1 + (r0), [XBLOCK, RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.load(in_ptr1 + (tl.broadcast_to(6 + ((-1) + ((-3) + r0)), [XBLOCK, RBLOCK])), rmask & tmp21, eviction_policy='evict_last', other=0.0)
    tmp74 = tl.where(tmp14, tmp72, tmp73)
    tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
    tmp76 = tl.where(tmp9, tmp74, tmp75)
    tmp77 = tl.load(in_ptr1 + (tl.broadcast_to(8 + ((-6) + r0), [XBLOCK, RBLOCK])), rmask & tmp35, eviction_policy='evict_last', other=0.0)
    tmp80 = tl.where(tmp34, tmp77, tmp79)
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp29, tmp80, tmp81)
    tmp83 = tl.load(in_ptr1 + (tl.broadcast_to(12 + ((-9) + r0), [XBLOCK, RBLOCK])), rmask & tmp46, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.where(tmp29, tmp82, tmp83)
    tmp85 = tl.where(tmp9, tmp76, tmp84)
    tmp86 = tl.where(tmp4, tmp70, tmp85)
    tmp87 = tmp86 * tmp53
    tmp88 = tl_math.exp(tmp87)
    tmp89 = tmp88 + tmp56
    tmp90 = tmp86 * tmp58
    tmp91 = tl_math.exp(tmp90)
    tmp92 = tmp89 + tmp91
    tmp93 = tmp86 * tmp62
    tmp94 = tl_math.exp(tmp93)
    tmp95 = tmp92 + tmp94
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK, RBLOCK])
    tmp98 = tl.where(rmask, tmp96, 0)
    tmp99 = tl.sum(tmp98, 1)[:, None]
    tmp100 = 12.0
    tmp101 = tmp69 / tmp100
    tmp102 = tmp99 / tmp100
    tmp103 = tmp101 + tmp102
    tmp106 = 16.0
    tmp107 = tmp105 / tmp106
    tmp108 = 2.0
    tmp109 = tmp107 * tmp108
    tmp110 = tmp103 - tmp109
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp110, None)
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
        buf8 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [addmm__2], Original ATen: [aten.addmm]
        extern_kernels.mm(arg0_1, reinterpret_tensor(arg1_1, (4, 4), (1, 4), 0), out=buf8)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [distmat, distmat_1, distmat_2, addmm__2, mul_6, exp_6, K_6, mul_7, exp_7, K_7, mul_8, exp_8, K_8, mean_2], Original ATen: [aten.add, aten.addmm, aten.mul, aten.exp, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_exp_mean_mul_0.run(buf9, arg0_1, arg1_1, buf0, buf4, buf10, 1, 16, grid=grid(1), stream=stream0)
        buf1 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [distmat, addmm_], Original ATen: [aten.add, aten.addmm]
        extern_kernels.addmm(buf0, arg0_1, reinterpret_tensor(arg0_1, (4, 4), (1, 4), 0), alpha=-2, beta=1, out=buf1)
        del arg0_1
        buf5 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [distmat_1, addmm__1], Original ATen: [aten.add, aten.addmm]
        extern_kernels.addmm(buf4, arg1_1, reinterpret_tensor(arg1_1, (4, 4), (1, 4), 0), alpha=-2, beta=1, out=buf5)
        del arg1_1
        del buf4
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf11 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [d_xx, mul, exp, K, mul_1, exp_1, K_1, mul_2, exp_2, K_2, mean, d_yy, mul_3, exp_3, K_3, mul_4, exp_4, K_4, mul_5, exp_5, K_5, mean_1, add_6, mul_6, exp_6, K_6, mul_7, exp_7, K_7, mul_8, exp_8, K_8, mean_2, mul_9, sub], Original ATen: [aten.stack, aten.mul, aten.exp, aten.add, aten.mean, aten.sub]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_exp_mean_mul_stack_sub_1.run(buf11, buf1, buf5, buf10, 1, 12, grid=grid(1), stream=stream0)
        del buf1
        del buf10
        del buf5
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
