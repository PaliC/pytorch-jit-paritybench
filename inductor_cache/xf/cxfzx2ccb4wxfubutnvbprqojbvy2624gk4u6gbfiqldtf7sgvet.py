# AOT ID: ['37_inference']
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


# kernel path: inductor_cache/bi/cbidvvppow55ovsd3ldirnuri4ao4mv2bleubtzbebhqnb6l54gp.py
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
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_0(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 - tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = tmp4 - tmp6
    tmp8 = r0
    tmp9 = tmp8.to(tl.int16)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12, tmp13, = triton_helpers.sort_with_index(tmp10, tmp11, None, 0, stable=False, descending=True)
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tl.full([RBLOCK], 256, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert((0 <= tmp18) & (tmp18 < 256), "index out of bounds: 0 <= tmp18 < 256")
    tmp20 = tl.load(in_ptr1 + (tmp18), None, eviction_policy='evict_last')
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tmp20.to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp26, = tl.associative_scan((tmp25,), 0, _triton_helper_fn_add0)
    tmp27 = tmp4 - tmp20
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp30, = tl.associative_scan((tmp29,), 0, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp12, None)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp26, None)
    tl.store(out_ptr4 + (tl.broadcast_to(r0, [RBLOCK])), tmp30, None)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/7o/c7ozq2psro2uaxn24cezpycrpv2xhuzc23kb5icmg6jh74x4by7w.py
# Topologically Sorted Source Nodes: [intersection, union, truediv, jaccard, relu, sub_5, loss], Original ATen: [aten.sub, aten.add, aten.div, aten.rsub, aten.relu, aten.dot]
# Source node to ATen node mapping:
#   intersection => sub_2
#   jaccard => sub_4
#   loss => mul_2, sum_2
#   relu => relu
#   sub_5 => sub_5
#   truediv => div
#   union => add
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_1, %cumsum), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %cumsum_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, %add), kwargs = {})
#   %sub_4 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %div), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%getitem,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %slice_2), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%sub_4, %sub_5, 0, 1, 256), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %slice_scatter_default), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_2,), kwargs = {})
triton_per_fused_add_div_dot_relu_rsub_sub_1 = async_compile.triton('triton_per_fused_add_div_dot_relu_rsub_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_dot_relu_rsub_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_dot_relu_rsub_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp6 = tl.load(in_out_ptr0 + (0))
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp24 = tl.load(in_out_ptr0 + (0))
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp26 = tl.load(in_ptr1 + (r0), None)
    tmp28 = tl.load(in_ptr2 + (r0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = r0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp8 = tl.load(in_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp5, other=0.0)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r0, [RBLOCK])), tmp5, other=0.0)
    tmp11 = tmp7 + tmp10
    tmp12 = tmp9 / tmp11
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tmp15 = tl.load(in_ptr1 + (tl.broadcast_to((-1) + r0, [RBLOCK])), tmp5, other=0.0)
    tmp16 = tmp7 - tmp15
    tmp17 = tl.load(in_ptr2 + (tl.broadcast_to((-1) + r0, [RBLOCK])), tmp5, other=0.0)
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
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp37, None)
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
        buf0 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf3 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf4 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul, signs, mul_1, errors, sort, gt_sorted, gts, cumsum, sub_3, cumsum_1], Original ATen: [aten.mul, aten.sub, aten.rsub, aten.sort, aten.index, aten.sum, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_index_mul_rsub_sort_sub_sum_0.run(arg0_1, arg1_1, buf0, buf2, buf3, buf4, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
        buf5 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [intersection, union, truediv, jaccard, relu, sub_5, loss], Original ATen: [aten.sub, aten.add, aten.div, aten.rsub, aten.relu, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_dot_relu_rsub_sub_1.run(buf5, buf0, buf3, buf4, 1, 256, grid=grid(1), stream=stream0)
        del buf0
        del buf3
        del buf4
    return (buf5, )


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
