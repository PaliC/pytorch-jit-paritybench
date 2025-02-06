# AOT ID: ['39_inference']
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


# kernel path: inductor_cache/rb/crb4eoqwd3wlz3jvgg6aezka343bwo7qr7oml6e4p6gbppyiamw6.py
# Topologically Sorted Source Nodes: [mul, signs, mul_1, errors, sort, gt_sorted, gts, cumsum, sub_3, cumsum_1], Original ATen: [aten.mul, aten.sub, aten.rsub, aten.sort, aten.index, aten.sum, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum => cumsum
#   cumsum_1 => cumsum_1
#   errors => sub_1
#   gt_sorted => index
#   gts => sum_1
#   mul => mul
#   mul_1 => mul_1
#   signs => sub
#   sort => sort
#   sub_3 => sub_3
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 2.0), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 1.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sub), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %mul_1), kwargs = {})
#   %sort : [num_users=2] = call_function[target=torch.ops.aten.sort.default](args = (%sub_1, 0, True), kwargs = {})
#   %index : [num_users=3] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_1, [%getitem_1]), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%index,), kwargs = {})
#   %cumsum : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%index, 0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %index), kwargs = {})
#   %cumsum_1 : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%sub_3, 0), kwargs = {})
triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_0 = async_compile.triton('triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_0', '''
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
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_0(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (64*(r0 // 16) + ((r0 % 16))), None)
    tmp1 = tl.load(in_ptr1 + (64*(r0 // 16) + ((r0 % 16))), None)
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 - tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = tmp4 - tmp6
    tmp8 = r0
    tmp9 = tmp8.to(tl.int16)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12, tmp13, = triton_helpers.sort_with_index(tmp10, tmp11, None, 1, stable=False, descending=True)
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tl.full([XBLOCK, RBLOCK], 64, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert((0 <= tmp18) & (tmp18 < 64), "index out of bounds: 0 <= tmp18 < 64")
    tmp20 = tl.load(in_ptr1 + (64*(((tmp18 // 16) % 4)) + ((tmp18 % 16))), None, eviction_policy='evict_last')
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp20.to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp26, = tl.associative_scan((tmp25,), 1, _triton_helper_fn_add0)
    tmp27 = tmp4 - tmp20
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp30, = tl.associative_scan((tmp29,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp12, None)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp26, None)
    tl.store(out_ptr4 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp30, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/ej/cejjh7l4i2n6zuci5cpfkupo3rc6f7b5wfpjkp6xslajk6ifraf3.py
# Topologically Sorted Source Nodes: [intersection, union, truediv, jaccard, relu, sub_5, loss, stack], Original ATen: [aten.sub, aten.add, aten.div, aten.rsub, aten.relu, aten.dot, aten.stack]
# Source node to ATen node mapping:
#   intersection => sub_2
#   jaccard => sub_4
#   loss => mul_2, sum_2
#   relu => relu
#   stack => cat
#   sub_5 => sub_5
#   truediv => div
#   union => add
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_1, %cumsum), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %cumsum_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, %add), kwargs = {})
#   %sub_4 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %div), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%getitem,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_3, %slice_4), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%sub_4, %sub_5, 0, 1, 64), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %slice_scatter_default), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_2,), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2, %unsqueeze_3],), kwargs = {})
triton_per_fused_add_div_dot_relu_rsub_stack_sub_1 = async_compile.triton('triton_per_fused_add_div_dot_relu_rsub_stack_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_dot_relu_rsub_stack_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_dot_relu_rsub_stack_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp6 = tl.load(in_out_ptr0 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_out_ptr0 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp26 = tl.load(in_ptr1 + (r0), None)
    tmp28 = tl.load(in_ptr2 + (r0), None)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = r0
    tmp4 = tl.full([1, 1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp8 = tl.load(in_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp5, other=0.0)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp5, other=0.0)
    tmp11 = tmp7 + tmp10
    tmp12 = tmp9 / tmp11
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tmp15 = tl.load(in_ptr1 + (tl.broadcast_to((-1) + r0, [XBLOCK, RBLOCK])), tmp5, other=0.0)
    tmp16 = tmp7 - tmp15
    tmp17 = tl.load(in_ptr2 + (tl.broadcast_to((-1) + r0, [XBLOCK, RBLOCK])), tmp5, other=0.0)
    tmp18 = tmp7 + tmp17
    tmp19 = tmp16 / tmp18
    tmp20 = tmp13 - tmp19
    tmp21 = tmp14 - tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tmp27 = tmp25 - tmp26
    tmp29 = tmp25 + tmp28
    tmp30 = tmp27 / tmp29
    tmp31 = 1.0
    tmp32 = tmp31 - tmp30
    tmp33 = tl.where(tmp5, tmp23, tmp32)
    tmp34 = tmp2 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.sum(tmp35, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/xt/cxtiegln6xkum57ozsiuv6fnnj4itfsc2puifyknloelgx2ba6ms.py
# Topologically Sorted Source Nodes: [mul_2, signs_1, mul_3, errors_1, sort_1, gt_sorted_1, gts_1, cumsum_2, sub_9, cumsum_3], Original ATen: [aten.mul, aten.sub, aten.rsub, aten.sort, aten.index, aten.sum, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum_2 => cumsum_2
#   cumsum_3 => cumsum_3
#   errors_1 => sub_7
#   gt_sorted_1 => index_1
#   gts_1 => sum_3
#   mul_2 => mul_3
#   mul_3 => mul_4
#   signs_1 => sub_6
#   sort_1 => sort_1
#   sub_9 => sub_9
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, 2.0), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, 1.0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %sub_6), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %mul_4), kwargs = {})
#   %sort_1 : [num_users=2] = call_function[target=torch.ops.aten.sort.default](args = (%sub_7, 0, True), kwargs = {})
#   %index_1 : [num_users=3] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_3, [%getitem_3]), kwargs = {})
#   %sum_3 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%index_1,), kwargs = {})
#   %cumsum_2 : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%index_1, 0), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %index_1), kwargs = {})
#   %cumsum_3 : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%sub_9, 0), kwargs = {})
triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_2 = async_compile.triton('triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_2', '''
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
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_2(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (16 + 64*(r0 // 16) + ((r0 % 16))), None)
    tmp1 = tl.load(in_ptr1 + (16 + 64*(r0 // 16) + ((r0 % 16))), None)
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 - tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = tmp4 - tmp6
    tmp8 = r0
    tmp9 = tmp8.to(tl.int16)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12, tmp13, = triton_helpers.sort_with_index(tmp10, tmp11, None, 1, stable=False, descending=True)
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tl.full([XBLOCK, RBLOCK], 64, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert((0 <= tmp18) & (tmp18 < 64), "index out of bounds: 0 <= tmp18 < 64")
    tmp20 = tl.load(in_ptr1 + (16 + 64*(((tmp18 // 16) % 4)) + ((tmp18 % 16))), None, eviction_policy='evict_last')
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp20.to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp26, = tl.associative_scan((tmp25,), 1, _triton_helper_fn_add0)
    tmp27 = tmp4 - tmp20
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp30, = tl.associative_scan((tmp29,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp12, None)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp26, None)
    tl.store(out_ptr4 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp30, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cikxj4dk5on3psenltlynd2irvirhj3qqaor6qnykggi2nelm25u.py
# Topologically Sorted Source Nodes: [intersection_1, union_1, truediv_1, jaccard_1, relu_1, sub_11, loss_1, stack], Original ATen: [aten.sub, aten.add, aten.div, aten.rsub, aten.relu, aten.dot, aten.stack]
# Source node to ATen node mapping:
#   intersection_1 => sub_8
#   jaccard_1 => sub_10
#   loss_1 => mul_5, sum_4
#   relu_1 => relu_1
#   stack => cat
#   sub_11 => sub_11
#   truediv_1 => div_1
#   union_1 => add_1
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_3, %cumsum_2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, %cumsum_3), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_8, %add_1), kwargs = {})
#   %sub_10 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %div_1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%getitem_2,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_9, %slice_10), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%sub_10, %sub_11, 0, 1, 64), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_1, %slice_scatter_default_1), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_5,), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2, %unsqueeze_3],), kwargs = {})
triton_per_fused_add_div_dot_relu_rsub_stack_sub_3 = async_compile.triton('triton_per_fused_add_div_dot_relu_rsub_stack_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_dot_relu_rsub_stack_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_dot_relu_rsub_stack_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp6 = tl.load(in_out_ptr0 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_out_ptr0 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp26 = tl.load(in_ptr1 + (r0), None)
    tmp28 = tl.load(in_ptr2 + (r0), None)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = r0
    tmp4 = tl.full([1, 1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp8 = tl.load(in_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp5, other=0.0)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp5, other=0.0)
    tmp11 = tmp7 + tmp10
    tmp12 = tmp9 / tmp11
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tmp15 = tl.load(in_ptr1 + (tl.broadcast_to((-1) + r0, [XBLOCK, RBLOCK])), tmp5, other=0.0)
    tmp16 = tmp7 - tmp15
    tmp17 = tl.load(in_ptr2 + (tl.broadcast_to((-1) + r0, [XBLOCK, RBLOCK])), tmp5, other=0.0)
    tmp18 = tmp7 + tmp17
    tmp19 = tmp16 / tmp18
    tmp20 = tmp13 - tmp19
    tmp21 = tmp14 - tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tmp27 = tmp25 - tmp26
    tmp29 = tmp25 + tmp28
    tmp30 = tmp27 / tmp29
    tmp31 = 1.0
    tmp32 = tmp31 - tmp30
    tmp33 = tl.where(tmp5, tmp23, tmp32)
    tmp34 = tmp2 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.sum(tmp35, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/vt/cvt2zgjdzhgconykq7z7pwnaytzzsxx6i66wkst63mr3foene47m.py
# Topologically Sorted Source Nodes: [mul_4, signs_2, mul_5, errors_2, sort_2, gt_sorted_2, gts_2, cumsum_4, sub_15, cumsum_5], Original ATen: [aten.mul, aten.sub, aten.rsub, aten.sort, aten.index, aten.sum, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum_4 => cumsum_4
#   cumsum_5 => cumsum_5
#   errors_2 => sub_13
#   gt_sorted_2 => index_2
#   gts_2 => sum_5
#   mul_4 => mul_6
#   mul_5 => mul_7
#   signs_2 => sub_12
#   sort_2 => sort_2
#   sub_15 => sub_15
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, 2.0), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_6, 1.0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %sub_12), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %mul_7), kwargs = {})
#   %sort_2 : [num_users=2] = call_function[target=torch.ops.aten.sort.default](args = (%sub_13, 0, True), kwargs = {})
#   %index_2 : [num_users=3] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_5, [%getitem_5]), kwargs = {})
#   %sum_5 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%index_2,), kwargs = {})
#   %cumsum_4 : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%index_2, 0), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %index_2), kwargs = {})
#   %cumsum_5 : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%sub_15, 0), kwargs = {})
triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_4 = async_compile.triton('triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_4', '''
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
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_4(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (32 + 64*(r0 // 16) + ((r0 % 16))), None)
    tmp1 = tl.load(in_ptr1 + (32 + 64*(r0 // 16) + ((r0 % 16))), None)
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 - tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = tmp4 - tmp6
    tmp8 = r0
    tmp9 = tmp8.to(tl.int16)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12, tmp13, = triton_helpers.sort_with_index(tmp10, tmp11, None, 1, stable=False, descending=True)
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tl.full([XBLOCK, RBLOCK], 64, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert((0 <= tmp18) & (tmp18 < 64), "index out of bounds: 0 <= tmp18 < 64")
    tmp20 = tl.load(in_ptr1 + (32 + 64*(((tmp18 // 16) % 4)) + ((tmp18 % 16))), None, eviction_policy='evict_last')
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp20.to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp26, = tl.associative_scan((tmp25,), 1, _triton_helper_fn_add0)
    tmp27 = tmp4 - tmp20
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp30, = tl.associative_scan((tmp29,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp12, None)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp26, None)
    tl.store(out_ptr4 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp30, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/bg/cbgwn6c4thbkbme2uo2t2m4s2rpcaqeupbfwgcgqorit6gogaz2y.py
# Topologically Sorted Source Nodes: [mul_6, signs_3, mul_7, errors_3, sort_3, gt_sorted_3, gts_3, cumsum_6, sub_21, cumsum_7], Original ATen: [aten.mul, aten.sub, aten.rsub, aten.sort, aten.index, aten.sum, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum_6 => cumsum_6
#   cumsum_7 => cumsum_7
#   errors_3 => sub_19
#   gt_sorted_3 => index_3
#   gts_3 => sum_7
#   mul_6 => mul_9
#   mul_7 => mul_10
#   signs_3 => sub_18
#   sort_3 => sort_3
#   sub_21 => sub_21
# Graph fragment:
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, 2.0), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, 1.0), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %sub_18), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %mul_10), kwargs = {})
#   %sort_3 : [num_users=2] = call_function[target=torch.ops.aten.sort.default](args = (%sub_19, 0, True), kwargs = {})
#   %index_3 : [num_users=3] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_7, [%getitem_7]), kwargs = {})
#   %sum_7 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%index_3,), kwargs = {})
#   %cumsum_6 : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%index_3, 0), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %index_3), kwargs = {})
#   %cumsum_7 : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%sub_21, 0), kwargs = {})
triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_5 = async_compile.triton('triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_5', '''
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
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_5(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (48 + 64*(r0 // 16) + ((r0 % 16))), None)
    tmp1 = tl.load(in_ptr1 + (48 + 64*(r0 // 16) + ((r0 % 16))), None)
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 - tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = tmp4 - tmp6
    tmp8 = r0
    tmp9 = tmp8.to(tl.int16)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12, tmp13, = triton_helpers.sort_with_index(tmp10, tmp11, None, 1, stable=False, descending=True)
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tl.full([XBLOCK, RBLOCK], 64, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert((0 <= tmp18) & (tmp18 < 64), "index out of bounds: 0 <= tmp18 < 64")
    tmp20 = tl.load(in_ptr1 + (48 + 64*(((tmp18 // 16) % 4)) + ((tmp18 % 16))), None, eviction_policy='evict_last')
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp20.to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp26, = tl.associative_scan((tmp25,), 1, _triton_helper_fn_add0)
    tmp27 = tmp4 - tmp20
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp30, = tl.associative_scan((tmp29,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp12, None)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp26, None)
    tl.store(out_ptr4 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp30, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/re/crermvq5bn7qpr6ugrehomhp7amnkxcc5cuvwvsn736b2ndqcajh.py
# Topologically Sorted Source Nodes: [loss_4], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   loss_4 => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%cat,), kwargs = {})
triton_poi_fused_mean_6 = async_compile.triton('triton_poi_fused_mean_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (1))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (2))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (3))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tmp7 = tmp4 + tmp6
    tmp10 = tmp7 + tmp9
    tmp11 = 4.0
    tmp12 = tmp10 / tmp11
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
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
        buf0 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf3 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf4 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul, signs, mul_1, errors, sort, gt_sorted, gts, cumsum, sub_3, cumsum_1], Original ATen: [aten.mul, aten.sub, aten.rsub, aten.sort, aten.index, aten.sum, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_0.run(arg0_1, arg1_1, buf0, buf2, buf3, buf4, 1, 64, grid=grid(1), stream=stream0)
        buf20 = buf2; del buf2  # reuse
        buf28 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf24 = reinterpret_tensor(buf28, (1, ), (1, ), 0)  # alias
        # Topologically Sorted Source Nodes: [intersection, union, truediv, jaccard, relu, sub_5, loss, stack], Original ATen: [aten.sub, aten.add, aten.div, aten.rsub, aten.relu, aten.dot, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_dot_relu_rsub_stack_sub_1.run(buf20, buf0, buf3, buf4, buf24, 1, 64, grid=grid(1), stream=stream0)
        buf5 = buf4; del buf4  # reuse
        buf7 = buf20; del buf20  # reuse
        buf8 = buf3; del buf3  # reuse
        buf9 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul_2, signs_1, mul_3, errors_1, sort_1, gt_sorted_1, gts_1, cumsum_2, sub_9, cumsum_3], Original ATen: [aten.mul, aten.sub, aten.rsub, aten.sort, aten.index, aten.sum, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_2.run(arg0_1, arg1_1, buf5, buf7, buf8, buf9, 1, 64, grid=grid(1), stream=stream0)
        buf21 = buf7; del buf7  # reuse
        buf25 = reinterpret_tensor(buf28, (1, ), (1, ), 1)  # alias
        # Topologically Sorted Source Nodes: [intersection_1, union_1, truediv_1, jaccard_1, relu_1, sub_11, loss_1, stack], Original ATen: [aten.sub, aten.add, aten.div, aten.rsub, aten.relu, aten.dot, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_dot_relu_rsub_stack_sub_3.run(buf21, buf5, buf8, buf9, buf25, 1, 64, grid=grid(1), stream=stream0)
        buf10 = buf9; del buf9  # reuse
        buf12 = buf21; del buf21  # reuse
        buf13 = buf8; del buf8  # reuse
        buf14 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [mul_4, signs_2, mul_5, errors_2, sort_2, gt_sorted_2, gts_2, cumsum_4, sub_15, cumsum_5], Original ATen: [aten.mul, aten.sub, aten.rsub, aten.sort, aten.index, aten.sum, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_4.run(arg0_1, arg1_1, buf10, buf12, buf13, buf14, 1, 64, grid=grid(1), stream=stream0)
        buf22 = buf12; del buf12  # reuse
        buf26 = reinterpret_tensor(buf28, (1, ), (1, ), 2)  # alias
        # Topologically Sorted Source Nodes: [intersection_2, union_2, truediv_2, jaccard_2, relu_2, sub_17, loss_2, stack], Original ATen: [aten.sub, aten.add, aten.div, aten.rsub, aten.relu, aten.dot, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_dot_relu_rsub_stack_sub_3.run(buf22, buf10, buf13, buf14, buf26, 1, 64, grid=grid(1), stream=stream0)
        buf15 = buf14; del buf14  # reuse
        buf17 = buf22; del buf22  # reuse
        buf18 = buf13; del buf13  # reuse
        buf19 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [mul_6, signs_3, mul_7, errors_3, sort_3, gt_sorted_3, gts_3, cumsum_6, sub_21, cumsum_7], Original ATen: [aten.mul, aten.sub, aten.rsub, aten.sort, aten.index, aten.sum, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_5.run(arg0_1, arg1_1, buf15, buf17, buf18, buf19, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
        buf23 = buf17; del buf17  # reuse
        buf27 = reinterpret_tensor(buf28, (1, ), (1, ), 3)  # alias
        # Topologically Sorted Source Nodes: [intersection_3, union_3, truediv_3, jaccard_3, relu_3, sub_23, loss_3, stack], Original ATen: [aten.sub, aten.add, aten.div, aten.rsub, aten.relu, aten.dot, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_dot_relu_rsub_stack_sub_3.run(buf23, buf15, buf18, buf19, buf27, 1, 64, grid=grid(1), stream=stream0)
        del buf15
        del buf18
        del buf19
        buf29 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [loss_4], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_6.run(buf28, buf29, 1, grid=grid(1), stream=stream0)
        del buf24
        del buf25
        del buf26
        del buf27
        del buf28
    return (buf29, )


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
