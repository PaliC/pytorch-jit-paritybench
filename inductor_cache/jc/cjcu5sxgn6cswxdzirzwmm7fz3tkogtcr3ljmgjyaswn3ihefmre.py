# AOT ID: ['5_forward']
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


# kernel path: inductor_cache/ue/cuejrkslrjd2erzme34tstnpwqvfbnkjsttwotphjk5youa6wmx3.py
# Topologically Sorted Source Nodes: [latents, flat_latents], Original ATen: [aten.clone, aten.view]
# Source node to ATen node mapping:
#   flat_latents => view
#   latents => clone
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [-1, 4]), kwargs = {})
triton_poi_fused_clone_view_0 = async_compile.triton('triton_poi_fused_clone_view_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_view_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (16*x1 + 64*(y0 // 16) + ((y0 % 16))), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 4*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/qk/cqk644sk3zr7pjmqaaenk33fqi6mgdohqvdzd32c72cjzzuhf5gn.py
# Topologically Sorted Source Nodes: [pow_1, sum_1, pow_2, sum_2, add, mul, dist], Original ATen: [aten.pow, aten.sum, aten.add, aten.mul, aten.sub]
# Source node to ATen node mapping:
#   add => add
#   dist => sub
#   mul => mul
#   pow_1 => pow_1
#   pow_2 => pow_2
#   sum_1 => sum_1
#   sum_2 => sum_2
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_2, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [1]), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %sum_2), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mul), kwargs = {})
triton_poi_fused_add_mul_pow_sub_sum_1 = async_compile.triton('triton_poi_fused_add_mul_pow_sub_sum_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fx/cfx226r3uxlm3mkbp6dv5on76xtl6xr2fhsfnu23xq7saafbu6tm.py
# Topologically Sorted Source Nodes: [argmin], Original ATen: [aten.argmin]
# Source node to ATen node mapping:
#   argmin => argmin
# Graph fragment:
#   %argmin : [num_users=1] = call_function[target=torch.ops.aten.argmin.default](args = (%sub, 1), kwargs = {})
triton_poi_fused_argmin_2 = async_compile.triton('triton_poi_fused_argmin_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_argmin_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_argmin_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 < tmp1
    tmp3 = tmp0 == tmp1
    tmp4 = tmp0 != tmp0
    tmp5 = tmp1 != tmp1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp2 | tmp6
    tmp8 = tmp4 & tmp5
    tmp9 = tmp3 | tmp8
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tmp7 | tmp13
    tmp15 = tl.where(tmp14, tmp0, tmp1)
    tmp16 = tl.where(tmp14, tmp10, tmp11)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp15 == tmp17
    tmp20 = tmp15 != tmp15
    tmp21 = tmp17 != tmp17
    tmp22 = tmp20 > tmp21
    tmp23 = tmp18 | tmp22
    tmp24 = tmp20 & tmp21
    tmp25 = tmp19 | tmp24
    tmp26 = tl.full([1], 2, tl.int64)
    tmp27 = tmp16 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tmp23 | tmp28
    tmp30 = tl.where(tmp29, tmp15, tmp17)
    tmp31 = tl.where(tmp29, tmp16, tmp26)
    tmp33 = tmp30 < tmp32
    tmp34 = tmp30 == tmp32
    tmp35 = tmp30 != tmp30
    tmp36 = tmp32 != tmp32
    tmp37 = tmp35 > tmp36
    tmp38 = tmp33 | tmp37
    tmp39 = tmp35 & tmp36
    tmp40 = tmp34 | tmp39
    tmp41 = tl.full([1], 3, tl.int64)
    tmp42 = tmp31 < tmp41
    tmp43 = tmp40 & tmp42
    tmp44 = tmp38 | tmp43
    tmp45 = tl.where(tmp44, tmp30, tmp32)
    tmp46 = tl.where(tmp44, tmp31, tmp41)
    tl.store(out_ptr0 + (x0), tmp46, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rb/crbjzljtzbiiilnm74ae7ol432szjykxxrzvxat3vkrai2q3dlvl.py
# Topologically Sorted Source Nodes: [scatter_], Original ATen: [aten.scatter]
# Source node to ATen node mapping:
#   scatter_ => scatter_upon_const_tensor
# Graph fragment:
#   %scatter_upon_const_tensor : [num_users=2] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [64, 4], background_val: 0, dtype: torch.float32, dim: 1, selector: %unsqueeze, val: 1})
triton_poi_fused_scatter_3 = async_compile.triton('triton_poi_fused_scatter_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scatter_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vx/cvxaxzxgp5yzerpmikaechyxpdoxqrgugk74y4ptiiitq3jhpdqa.py
# Topologically Sorted Source Nodes: [latents, commitment_loss, mul_1, vq_loss], Original ATen: [aten.clone, aten.mse_loss, aten.mul, aten.add]
# Source node to ATen node mapping:
#   commitment_loss => mean, pow_3, sub_1
#   latents => clone
#   mul_1 => mul_1
#   vq_loss => add_1
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_1 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %clone), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 0.25), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mean), kwargs = {})
triton_per_fused_add_clone_mse_loss_mul_4 = async_compile.triton('triton_per_fused_add_clone_mse_loss_mul_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_mse_loss_mul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clone_mse_loss_mul_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
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
    r3 = rindex
    r0 = (rindex % 4)
    r1 = ((rindex // 4) % 16)
    r2 = rindex // 64
    tmp0 = tl.load(in_ptr0 + (r3), None)
    tmp1 = tl.load(in_ptr1 + (r1 + 16*r0 + 64*r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.25
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 + tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp11, None)
''', device_str='cuda')


# kernel path: inductor_cache/f3/cf3mbvroqkpv6drf4jkxicpdbbtowgmjwbwegpuve7ucsmr3yhra.py
# Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (y0 + 4*x2 + 64*y1), xmask & ymask)
    tmp2 = tmp1 - tmp0
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x2 + 16*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/nw/cnw75sbkhrgrv3w26hu7fnsfllrqrw5ieumtedzyfcywexvfvby2.py
# Topologically Sorted Source Nodes: [latents, commitment_loss], Original ATen: [aten.clone, aten.mse_loss, aten.mse_loss_backward]
# Source node to ATen node mapping:
#   commitment_loss => sub_1
#   latents => clone
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_1 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %clone), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, 0.0078125), kwargs = {})
triton_poi_fused_clone_mse_loss_mse_loss_backward_6 = async_compile.triton('triton_poi_fused_clone_mse_loss_mse_loss_backward_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mse_loss_mse_loss_backward_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mse_loss_mse_loss_backward_6(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + 16*x2 + 64*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = 0.0078125
    tmp4 = tmp2 * tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 4*y3), tmp4, xmask & ymask)
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
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [latents, flat_latents], Original ATen: [aten.clone, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_view_0.run(primals_1, buf0, 64, 4, grid=grid(64, 4), stream=stream0)
        buf1 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf1)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [pow_1, sum_1, pow_2, sum_2, add, mul, dist], Original ATen: [aten.pow, aten.sum, aten.add, aten.mul, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sub_sum_1.run(buf2, buf0, primals_2, 256, grid=grid(256), stream=stream0)
        buf3 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [argmin], Original ATen: [aten.argmin]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmin_2.run(buf2, buf3, 64, grid=grid(64), stream=stream0)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [scatter_], Original ATen: [aten.scatter]
        stream0 = get_raw_stream(0)
        triton_poi_fused_scatter_3.run(buf3, buf4, 256, grid=grid(256), stream=stream0)
        del buf3
        buf5 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [quantized_latents], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, primals_2, out=buf5)
        del primals_2
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf9 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [latents, commitment_loss, mul_1, vq_loss], Original ATen: [aten.clone, aten.mse_loss, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_mse_loss_mul_4.run(buf9, buf5, primals_1, 1, 256, grid=grid(1), stream=stream0)
        buf7 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(primals_1, buf5, buf7, 16, 16, grid=grid(16, 16), stream=stream0)
        buf8 = reinterpret_tensor(buf5, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [latents, commitment_loss], Original ATen: [aten.clone, aten.mse_loss, aten.mse_loss_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mse_loss_mse_loss_backward_6.run(buf8, primals_1, 64, 4, grid=grid(64, 4), stream=stream0)
        del primals_1
    return (buf7, buf9, buf8, reinterpret_tensor(buf4, (4, 64), (1, 4), 0), )


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
