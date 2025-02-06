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


# kernel path: inductor_cache/xk/cxkedajjrqtlo3ewivrhl2jcpauj6eznfg5epwkijjbqeihdqylj.py
# Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   scores_6 => clone_2, clone_3
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton('triton_poi_fused_clone_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/qf/cqfkpja72jsdrksdvz7pc5yyuzromuhqbg2hwhgzxo6r5qq7cr3x.py
# Topologically Sorted Source Nodes: [scores_7, repeat, scores_8, sub_1, attention_mask_extended_add_1, scores_9, scores_10], Original ATen: [aten.div, aten.repeat, aten.mul, aten.rsub, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attention_mask_extended_add_1 => mul_6
#   repeat => repeat
#   scores_10 => amax_1, exp_1, sub_3, sum_5
#   scores_7 => div_2
#   scores_8 => mul_5
#   scores_9 => add_1
#   sub_1 => sub_2
# Graph fragment:
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_7, 1.0), kwargs = {})
#   %repeat : [num_users=5] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze_2, [1, 4, 1, 1]), kwargs = {})
#   %mul_5 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %repeat), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %repeat), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, -10000.0), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_1, [-1], True), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %amax_1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
triton_poi_fused__softmax_add_div_mul_repeat_rsub_1 = async_compile.triton('triton_poi_fused__softmax_add_div_mul_repeat_rsub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_div_mul_repeat_rsub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_div_mul_repeat_rsub_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (4*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (4*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (1 + 4*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (2 + 4*x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr1 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (3 + 4*x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tmp1 - tmp5
    tmp8 = -10000.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tmp11 * tmp1
    tmp14 = tmp3 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp1 - tmp14
    tmp17 = tmp16 * tmp8
    tmp18 = tmp15 + tmp17
    tmp19 = triton_helpers.maximum(tmp10, tmp18)
    tmp21 = tmp20 * tmp1
    tmp23 = tmp3 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp1 - tmp23
    tmp26 = tmp25 * tmp8
    tmp27 = tmp24 + tmp26
    tmp28 = triton_helpers.maximum(tmp19, tmp27)
    tmp30 = tmp29 * tmp1
    tmp32 = tmp3 * tmp31
    tmp33 = tmp30 * tmp32
    tmp34 = tmp1 - tmp32
    tmp35 = tmp34 * tmp8
    tmp36 = tmp33 + tmp35
    tmp37 = triton_helpers.maximum(tmp28, tmp36)
    tmp38 = tmp10 - tmp37
    tmp39 = tl_math.exp(tmp38)
    tmp40 = tmp18 - tmp37
    tmp41 = tl_math.exp(tmp40)
    tmp42 = tmp39 + tmp41
    tmp43 = tmp27 - tmp37
    tmp44 = tl_math.exp(tmp43)
    tmp45 = tmp42 + tmp44
    tmp46 = tmp36 - tmp37
    tmp47 = tl_math.exp(tmp46)
    tmp48 = tmp45 + tmp47
    tl.store(out_ptr0 + (x3), tmp37, xmask)
    tl.store(out_ptr1 + (x3), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dk/cdkp2r6jj3ic7ubaxndpgidvxa5jznaszeov3l53e7wsm3cujfh5.py
# Topologically Sorted Source Nodes: [scores_7, repeat, scores_8, sub_1, attention_mask_extended_add_1, scores_9, scores_10, scores_11, mse_loss, sum_7], Original ATen: [aten.div, aten.repeat, aten.mul, aten.rsub, aten.add, aten._softmax, aten.mse_loss, aten.sum]
# Source node to ATen node mapping:
#   attention_mask_extended_add_1 => mul_6
#   mse_loss => pow_1, sub_4, sum_9
#   repeat => repeat
#   scores_10 => div_3, exp_1, sub_3
#   scores_11 => mul_7
#   scores_7 => div_2
#   scores_8 => mul_5
#   scores_9 => add_1
#   sub_1 => sub_2
#   sum_7 => sum_10
# Graph fragment:
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_7, 1.0), kwargs = {})
#   %repeat : [num_users=5] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze_2, [1, 4, 1, 1]), kwargs = {})
#   %mul_5 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %repeat), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %repeat), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, -10000.0), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %amax_1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_5), kwargs = {})
#   %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %repeat), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_8, %view_9), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_1,), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%repeat,), kwargs = {})
triton_per_fused__softmax_add_div_mse_loss_mul_repeat_rsub_sum_2 = async_compile.triton('triton_per_fused__softmax_add_div_mse_loss_mul_repeat_rsub_sum_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_div_mse_loss_mul_repeat_rsub_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_div_mse_loss_mul_repeat_rsub_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    r4 = rindex
    r1 = ((rindex // 4) % 4)
    r3 = rindex // 64
    r0 = (rindex % 4)
    r6 = rindex // 4
    tmp0 = tl.load(in_ptr0 + (r4), None)
    tmp3 = tl.load(in_ptr1 + (r1 + 4*r3), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (r0 + 4*r3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (r6), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (r6), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (r4), None)
    tmp19 = tl.load(in_ptr1 + (4*(r4 // 64) + (((r4 // 4) % 4))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (4*(r4 // 64) + ((r4 % 4))), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tmp1 - tmp5
    tmp8 = -10000.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tmp10 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp15 = tmp13 / tmp14
    tmp16 = tmp15 * tmp5
    tmp18 = tmp17 * tmp1
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 * tmp21
    tmp23 = tmp2 * tmp21
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tl.store(out_ptr0 + (tl.broadcast_to(r4, [RBLOCK])), tmp16, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp28, None)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/ix/cixhd2rd2uwdoueftxhrkg7qc2ivu7mflotmwc4kfx7hecj4eaow.py
# Topologically Sorted Source Nodes: [sum_4, global_score_1], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   global_score_1 => sum_7
#   sum_4 => sum_6
# Graph fragment:
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_7, [2]), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sum_6, [1]), kwargs = {})
triton_poi_fused_sum_3 = async_compile.triton('triton_poi_fused_sum_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (4 + x0 + 64*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (8 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (12 + x0 + 64*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp8 = tl.load(in_ptr0 + (20 + x0 + 64*x1), xmask)
    tmp10 = tl.load(in_ptr0 + (24 + x0 + 64*x1), xmask)
    tmp12 = tl.load(in_ptr0 + (28 + x0 + 64*x1), xmask)
    tmp15 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp16 = tl.load(in_ptr0 + (36 + x0 + 64*x1), xmask)
    tmp18 = tl.load(in_ptr0 + (40 + x0 + 64*x1), xmask)
    tmp20 = tl.load(in_ptr0 + (44 + x0 + 64*x1), xmask)
    tmp23 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp24 = tl.load(in_ptr0 + (52 + x0 + 64*x1), xmask)
    tmp26 = tl.load(in_ptr0 + (56 + x0 + 64*x1), xmask)
    tmp28 = tl.load(in_ptr0 + (60 + x0 + 64*x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tmp6 + tmp13
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tmp14 + tmp21
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp22 + tmp29
    tl.store(out_ptr0 + (x2), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g4/cg4cni2fqqukztpckupxbngxv3kkytcahxpp24cj4p5evosd4grp.py
# Topologically Sorted Source Nodes: [mask_1], Original ATen: [aten.ones_like]
# Source node to ATen node mapping:
#   mask_1 => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_ones_like_4 = async_compile.triton('triton_poi_fused_ones_like_4', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_like_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_like_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fj/cfjezaz2jp33gxr6f5hylanzjr3aiudvsm7lqsvfdfiha5nyio6n.py
# Topologically Sorted Source Nodes: [mask_1, setitem_1], Original ATen: [aten.ones_like, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   mask_1 => full_default
#   setitem_1 => full_default_1, index_put_1
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [None, %lift_fresh_copy_4, %lift_fresh_copy_5], %full_default_1), kwargs = {})
triton_poi_fused_index_put_lift_fresh_ones_like_5 = async_compile.triton('triton_poi_fused_index_put_lift_fresh_ones_like_5', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_ones_like_5', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_put_lift_fresh_ones_like_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tmp7 = tl.full([1], 3, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.where(tmp8, tmp1, tmp7)
    tmp10 = tl.where(tmp2, tmp6, tmp9)
    tmp11 = 0.0
    tl.store(out_ptr0 + (5*tmp10 + 16*x1), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/36/c36alzkuixyj7r7us5iju6di4vexsyxrb677vu56nfhluaujhkco.py
# Topologically Sorted Source Nodes: [local_score_2, local_score_3], Original ATen: [aten.sum, aten.mul]
# Source node to ATen node mapping:
#   local_score_2 => sum_8
#   local_score_3 => mul_8
# Graph fragment:
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_7, [1]), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_8, %index_put_1), kwargs = {})
triton_poi_fused_mul_sum_6 = async_compile.triton('triton_poi_fused_mul_sum_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp7 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r4/cr46mgb5sukaj5km7t2sax6dh46atnububwvbrafdmdrgwru4zxx.py
# Topologically Sorted Source Nodes: [sd, sd_1, norm_sd, sd_2, sd_3, norm_sd_1], Original ATen: [aten.sub, aten.gather, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   norm_sd => pow_2, sum_11
#   norm_sd_1 => pow_4, sum_12
#   sd => sub_5
#   sd_1 => gather_3
#   sd_2 => sub_6
#   sd_3 => gather_7
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_7, %unsqueeze_8), kwargs = {})
#   %gather_3 : [num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%sub_5, 3, %expand_7), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%gather_3, 2), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [-1], True), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_23, %unsqueeze_24), kwargs = {})
#   %gather_7 : [num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%sub_6, 3, %expand_11), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%gather_7, 2), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_4, [-1], True), kwargs = {})
triton_poi_fused_gather_linalg_vector_norm_sub_7 = async_compile.triton('triton_poi_fused_gather_linalg_vector_norm_sub_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gather_linalg_vector_norm_sub_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gather_linalg_vector_norm_sub_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex // 16
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (x0 + 4*tmp4 + 16*x2), xmask)
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 4")
    tmp11 = tl.load(in_ptr2 + (4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp13 = tmp11 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.load(in_ptr2 + (1 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (1 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp17 = tmp15 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tmp14 + tmp18
    tmp20 = tl.load(in_ptr2 + (2 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (2 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp22 = tmp20 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tmp19 + tmp23
    tmp25 = tl.load(in_ptr2 + (3 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (3 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tmp24 + tmp28
    tmp30 = tl.load(in_ptr3 + (4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp32 = tmp30 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tl.load(in_ptr3 + (1 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr3 + (1 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tmp33 + tmp37
    tmp39 = tl.load(in_ptr3 + (2 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr3 + (2 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp41 = tmp39 - tmp40
    tmp42 = tmp41 * tmp41
    tmp43 = tmp38 + tmp42
    tmp44 = tl.load(in_ptr3 + (3 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr3 + (3 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp46 = tmp44 - tmp45
    tmp47 = tmp46 * tmp46
    tmp48 = tmp43 + tmp47
    tl.store(out_ptr0 + (x5), tmp29, xmask)
    tl.store(out_ptr1 + (x5), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2j/c2j34gyi7cxdvf5ju3pzntdlg4lnqwy4lvswyziwzssppp3sbpvu.py
# Topologically Sorted Source Nodes: [sd, sd_1, norm_sd, sd_2, sd_3, norm_sd_1], Original ATen: [aten.sub, aten.gather, aten.div]
# Source node to ATen node mapping:
#   norm_sd => div_5
#   norm_sd_1 => div_6
#   sd => sub_5
#   sd_1 => gather_3
#   sd_2 => sub_6
#   sd_3 => gather_7
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_7, %unsqueeze_8), kwargs = {})
#   %gather_3 : [num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%sub_5, 3, %expand_7), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%gather_3, %expand_8), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_23, %unsqueeze_24), kwargs = {})
#   %gather_7 : [num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%sub_6, 3, %expand_11), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%gather_7, %expand_12), kwargs = {})
triton_poi_fused_div_gather_sub_8 = async_compile.triton('triton_poi_fused_div_gather_sub_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_gather_sub_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_gather_sub_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex // 16
    x1 = ((xindex // 4) % 4)
    x3 = xindex // 64
    x0 = (xindex % 4)
    x7 = xindex // 4
    x8 = xindex
    tmp0 = tl.load(in_ptr0 + (x5), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x7), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x7), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (x1 + 4*tmp4 + 16*x3), xmask, eviction_policy='evict_last')
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 4")
    tmp11 = tl.load(in_ptr2 + (x0 + 4*tmp4 + 16*x3), xmask)
    tmp12 = tl.load(in_ptr2 + (x0 + 4*tmp9 + 16*x3), xmask)
    tmp13 = tmp11 - tmp12
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = 1e-12
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tmp13 / tmp17
    tmp19 = tl.load(in_ptr4 + (x0 + 4*tmp4 + 16*x3), xmask)
    tmp20 = tl.load(in_ptr4 + (x0 + 4*tmp9 + 16*x3), xmask)
    tmp21 = tmp19 - tmp20
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = triton_helpers.maximum(tmp23, tmp16)
    tmp25 = tmp21 / tmp24
    tl.store(out_ptr0 + (x8), tmp18, xmask)
    tl.store(out_ptr1 + (x8), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hp/chp45hqzaurituj7bfhj552xv447b6rb34two3xpfd3v3c6fumgg.py
# Topologically Sorted Source Nodes: [ne, mask_2], Original ATen: [aten.ne, aten._to_copy]
# Source node to ATen node mapping:
#   mask_2 => convert_element_type_1
#   ne => ne
# Graph fragment:
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_14, 0), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.float32), kwargs = {})
triton_poi_fused__to_copy_ne_9 = async_compile.triton('triton_poi_fused__to_copy_ne_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_ne_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_ne_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crzfvbisjdgwwdbh4frahlgei4c76ezrnb6466n3dkqwbrzbxzlf.py
# Topologically Sorted Source Nodes: [ne, mask_2, setitem_2], Original ATen: [aten.ne, aten._to_copy, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   mask_2 => convert_element_type_1
#   ne => ne
#   setitem_2 => full_default_2, index_put_2
# Graph fragment:
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_14, 0), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.float32), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_2 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%convert_element_type_1, [None, None, None, %lift_fresh_copy_7, %lift_fresh_copy_8], %full_default_2), kwargs = {})
triton_poi_fused__to_copy_index_put_lift_fresh_ne_10 = async_compile.triton('triton_poi_fused__to_copy_index_put_lift_fresh_ne_10', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_index_put_lift_fresh_ne_10', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_index_put_lift_fresh_ne_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tmp7 = tl.full([1], 3, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.where(tmp8, tmp1, tmp7)
    tmp10 = tl.where(tmp2, tmp6, tmp9)
    tmp11 = 0.0
    tl.store(out_ptr0 + (5*tmp10 + 16*x1), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7c/c7ceyd6fl4w2nlftcyoiqe2qnkiiydc3mwsmhfqwd57okrbkqu2e.py
# Topologically Sorted Source Nodes: [loss_pair, eq, attention_mask_extended_6, attention_mask_extended_7, smooth_l1_loss, sum_8, loss_triplet, add_4], Original ATen: [aten.div, aten.eq, aten._to_copy, aten.mul, aten.smooth_l1_loss, aten.sum, aten.add]
# Source node to ATen node mapping:
#   add_4 => add_4
#   attention_mask_extended_6 => convert_element_type
#   attention_mask_extended_7 => mul_10
#   eq => eq
#   loss_pair => div_4
#   loss_triplet => div_8
#   smooth_l1_loss => abs_1, div_7, lt, mul_15, pow_6, sub_7, sub_8, sum_13, where
#   sum_8 => sum_14
# Graph fragment:
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_9, %sum_10), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_15, 4), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.float32), kwargs = {})
#   %mul_10 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %index_put_2), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_22, %view_23), kwargs = {})
#   %abs_1 : [num_users=3] = call_function[target=torch.ops.aten.abs.default](args = (%sub_7,), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%abs_1, 2), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_6, 0.5), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_15, 1.0), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt, %div_7, %sub_8), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where,), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_10,), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_13, %sum_14), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %div_8), kwargs = {})
triton_red_fused__to_copy_add_div_eq_mul_smooth_l1_loss_sum_11 = async_compile.triton('triton_red_fused__to_copy_add_div_eq_mul_smooth_l1_loss_sum_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_eq_mul_smooth_l1_loss_sum_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_eq_mul_smooth_l1_loss_sum_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp67 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp70 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r6 = rindex // 16
        r1 = ((rindex // 4) % 4)
        r3 = rindex // 64
        r0 = (rindex % 4)
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (r6), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr3 + (r4), rmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_out_ptr0 + (r4), rmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr4 + (r4), rmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr0 + (r4 // 16), rmask, eviction_policy='evict_last', other=0.0)
        tmp53 = tl.load(in_ptr5 + (r4), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp4 < 4")
        tmp6 = tl.load(in_ptr1 + (r1 + 4*tmp4 + 16*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp6 + tmp1
        tmp8 = tmp6 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp6)
        tl.device_assert(((0 <= tmp9) & (tmp9 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp9 < 4")
        tmp11 = tl.load(in_ptr2 + (tmp4 + 4*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (tmp9 + 4*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.load(in_ptr1 + (r0 + 4*tmp4 + 16*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp14 + tmp1
        tmp16 = tmp14 < 0
        tmp17 = tl.where(tmp16, tmp15, tmp14)
        tl.device_assert(((0 <= tmp17) & (tmp17 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp17 < 4")
        tmp19 = tl.load(in_ptr2 + (tmp17 + 4*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp11 + tmp19
        tmp21 = tmp13 * tmp20
        tmp22 = 4.0
        tmp23 = tmp21 == tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp28 = tmp27 * tmp26
        tmp31 = tmp30 + tmp1
        tmp32 = tmp30 < 0
        tmp33 = tl.where(tmp32, tmp31, tmp30)
        tl.device_assert(((0 <= tmp33) & (tmp33 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp33 < 4")
        tmp35 = tl.load(in_ptr1 + (4*tmp33 + 16*(r4 // 64) + (((r4 // 4) % 4))), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp35 + tmp1
        tmp37 = tmp35 < 0
        tmp38 = tl.where(tmp37, tmp36, tmp35)
        tl.device_assert(((0 <= tmp38) & (tmp38 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp38 < 4")
        tmp40 = tl.load(in_ptr2 + (tmp33 + 4*(r4 // 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.load(in_ptr2 + (tmp38 + 4*(r4 // 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp42 = tmp40 + tmp41
        tmp43 = tl.load(in_ptr1 + (4*tmp33 + 16*(r4 // 64) + ((r4 % 4))), rmask, eviction_policy='evict_first', other=0.0)
        tmp44 = tmp43 + tmp1
        tmp45 = tmp43 < 0
        tmp46 = tl.where(tmp45, tmp44, tmp43)
        tl.device_assert(((0 <= tmp46) & (tmp46 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp46 < 4")
        tmp48 = tl.load(in_ptr2 + (tmp46 + 4*(r4 // 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tmp40 + tmp48
        tmp50 = tmp42 * tmp49
        tmp51 = tmp50 == tmp22
        tmp52 = tmp51.to(tl.float32)
        tmp54 = tmp52 * tmp53
        tmp55 = tmp29 * tmp54
        tmp56 = tmp28 - tmp55
        tmp57 = tl_math.abs(tmp56)
        tmp58 = 1.0
        tmp59 = tmp57 < tmp58
        tmp60 = tmp57 * tmp57
        tmp61 = 0.5
        tmp62 = tmp60 * tmp61
        tmp63 = tmp62 * tmp58
        tmp64 = tmp57 - tmp61
        tmp65 = tl.where(tmp59, tmp63, tmp64)
        tmp66 = tl.broadcast_to(tmp65, [XBLOCK, RBLOCK])
        tmp68 = _tmp67 + tmp66
        _tmp67 = tl.where(rmask, tmp68, _tmp67)
        tmp69 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp71 = _tmp70 + tmp69
        _tmp70 = tl.where(rmask, tmp71, _tmp70)
    tmp67 = tl.sum(_tmp67, 1)[:, None]
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    tmp72 = tl.load(in_out_ptr1 + (0))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, 1])
    tmp74 = tl.load(in_ptr6 + (0))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK, 1])
    tmp76 = tmp73 / tmp75
    tmp77 = tmp67 / tmp70
    tmp78 = tmp76 + tmp77
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp78, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4), (16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        buf1 = empty_strided_cuda((4, 4, 1, 4), (16, 4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(arg2_1, buf0, buf1, 16, 4, grid=grid(16, 4), stream=stream0)
        buf2 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (16, 4, 1), (4, 1, 0), 0), reinterpret_tensor(buf1, (16, 1, 4), (4, 0, 1), 0), out=buf2)
        buf3 = reinterpret_tensor(buf1, (4, 4, 4, 1), (16, 4, 1, 64), 0); del buf1  # reuse
        buf4 = reinterpret_tensor(buf0, (4, 4, 4, 1), (16, 4, 1, 64), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [scores_7, repeat, scores_8, sub_1, attention_mask_extended_add_1, scores_9, scores_10], Original ATen: [aten.div, aten.repeat, aten.mul, aten.rsub, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_div_mul_repeat_rsub_1.run(buf2, arg0_1, buf3, buf4, 64, grid=grid(64), stream=stream0)
        buf16 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        buf17 = empty_strided_cuda((4, 4, 1, 4), (16, 4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(arg1_1, buf16, buf17, 16, 4, grid=grid(16, 4), stream=stream0)
        buf18 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (16, 4, 1), (4, 1, 0), 0), reinterpret_tensor(buf17, (16, 1, 4), (4, 0, 1), 0), out=buf18)
        del buf16
        del buf17
        buf5 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf19 = empty_strided_cuda((), (), torch.float32)
        buf20 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [scores_7, repeat, scores_8, sub_1, attention_mask_extended_add_1, scores_9, scores_10, scores_11, mse_loss, sum_7], Original ATen: [aten.div, aten.repeat, aten.mul, aten.rsub, aten.add, aten._softmax, aten.mse_loss, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_div_mse_loss_mul_repeat_rsub_sum_2.run(buf2, arg0_1, buf3, buf4, buf18, buf5, buf19, buf20, 1, 256, grid=grid(1), stream=stream0)
        buf6 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sum_4, global_score_1], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_3.run(buf5, buf6, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [topk_2], Original ATen: [aten.topk]
        buf7 = torch.ops.aten.topk.default(buf6, 4, 1)
        del buf6
        buf9 = buf7[1]
        del buf7
        buf10 = reinterpret_tensor(buf4, (4, 4, 4), (16, 4, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [mask_1], Original ATen: [aten.ones_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_like_4.run(buf10, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [mask_1, setitem_1], Original ATen: [aten.ones_like, aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_put_lift_fresh_ones_like_5.run(buf10, 16, grid=grid(16), stream=stream0)
        buf12 = reinterpret_tensor(buf3, (4, 4, 4), (16, 4, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [local_score_2, local_score_3], Original ATen: [aten.sum, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sum_6.run(buf5, buf10, buf12, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [local_score_2, local_score_3, topk_3], Original ATen: [aten.sum, aten.mul, aten.topk]
        buf13 = torch.ops.aten.topk.default(buf12, 4, 2)
        buf15 = buf13[1]
        del buf13
        buf21 = reinterpret_tensor(buf12, (4, 1, 4, 4, 1), (16, 64, 4, 1, 64), 0); del buf12  # reuse
        buf27 = reinterpret_tensor(buf10, (4, 1, 4, 4, 1), (16, 64, 4, 1, 64), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [sd, sd_1, norm_sd, sd_2, sd_3, norm_sd_1], Original ATen: [aten.sub, aten.gather, aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gather_linalg_vector_norm_sub_7.run(buf9, buf15, arg1_1, arg2_1, buf21, buf27, 64, grid=grid(64), stream=stream0)
        buf22 = reinterpret_tensor(buf5, (4, 1, 4, 4, 4), (64, 64, 16, 4, 1), 0); del buf5  # reuse
        buf28 = reinterpret_tensor(buf2, (4, 1, 4, 4, 4), (64, 64, 16, 4, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [sd, sd_1, norm_sd, sd_2, sd_3, norm_sd_1], Original ATen: [aten.sub, aten.gather, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_gather_sub_8.run(buf9, buf15, arg1_1, buf21, arg2_1, buf27, buf22, buf28, 256, grid=grid(256), stream=stream0)
        del arg1_1
        del arg2_1
        del buf21
        del buf27
        buf23 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [angle], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf22, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf22, (16, 4, 4), (16, 1, 4), 0), out=buf23)
        buf24 = reinterpret_tensor(buf22, (4, 1, 4, 4, 4), (64, 256, 16, 4, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [ne, mask_2], Original ATen: [aten.ne, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_ne_9.run(buf23, buf24, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [ne, mask_2, setitem_2], Original ATen: [aten.ne, aten._to_copy, aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_index_put_lift_fresh_ne_10.run(buf24, 64, grid=grid(64), stream=stream0)
        buf29 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [angle_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf28, (16, 4, 4), (16, 1, 4), 0), out=buf29)
        buf30 = reinterpret_tensor(buf28, (4, 1, 4, 4, 4), (64, 256, 16, 4, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [ne_1, mask_3], Original ATen: [aten.ne, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_ne_9.run(buf29, buf30, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [ne_1, mask_3, setitem_3], Original ATen: [aten.ne, aten._to_copy, aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_index_put_lift_fresh_ne_10.run(buf30, 64, grid=grid(64), stream=stream0)
        buf32 = reinterpret_tensor(buf23, (256, ), (1, ), 0); del buf23  # reuse
        buf35 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [loss_pair, eq, attention_mask_extended_6, attention_mask_extended_7, smooth_l1_loss, sum_8, loss_triplet, add_4], Original ATen: [aten.div, aten.eq, aten._to_copy, aten.mul, aten.smooth_l1_loss, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_eq_mul_smooth_l1_loss_sum_11.run(buf32, buf35, buf9, buf15, arg0_1, buf24, buf29, buf30, buf20, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del buf15
        del buf20
        del buf24
        del buf29
        del buf30
        del buf32
        del buf9
    return (buf35, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
