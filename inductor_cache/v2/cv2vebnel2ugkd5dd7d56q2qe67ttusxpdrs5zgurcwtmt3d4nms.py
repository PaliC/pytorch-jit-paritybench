# AOT ID: ['4_inference']
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


# kernel path: inductor_cache/pb/cpbom3evnjfqrefhxkizt5falsw73r2624hmjcf6quiozsh5lg4f.py
# Topologically Sorted Source Nodes: [sub, sub_1, gt_area, sub_2, sub_3, pr_area, add_1, inter, union, gt_cent_x, pr_cent_x, sub_6, pow_1, gt_cent_y, pr_cent_y, sub_7, pow_2, cent_dis, lt_1, rb_1, sub_8, pow_3, diag_dis, add_3, reg], Original ATen: [aten.sub, aten.mul, aten.add, aten.mean, aten.pow, aten.minimum, aten.maximum, aten.sum, aten.div]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_3 => add_3
#   cent_dis => add_2
#   diag_dis => sum_1
#   gt_area => mul
#   gt_cent_x => mean
#   gt_cent_y => mean_1
#   inter => mul_2
#   lt_1 => minimum_1
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pr_area => mul_1
#   pr_cent_x => mean_2
#   pr_cent_y => mean_3
#   rb_1 => maximum_1
#   reg => div_1
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   union => sub_5
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select, %select_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_2, %select_3), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %sub_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_6, %select_7), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %sub_3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %select_9), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %mul_2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_20, [-1]), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_24, [-1]), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %mean_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_6, 2.0), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_22, [-1]), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_26, [-1]), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_1, %mean_3), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_7, 2.0), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %minimum_1 : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%slice_28, %slice_30), kwargs = {})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%slice_32, %slice_34), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum_1, %maximum_1), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [-1]), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-05), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_2, %add_3), kwargs = {})
triton_poi_fused_add_div_maximum_mean_minimum_mul_pow_sub_sum_0 = async_compile.triton('triton_poi_fused_add_div_maximum_mean_minimum_mul_pow_sub_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_maximum_mean_minimum_mul_pow_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_maximum_mean_minimum_mul_pow_sub_sum_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp2 - tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = tmp13 - tmp16
    tmp18 = tmp17 + tmp7
    tmp19 = triton_helpers.maximum(tmp18, tmp9)
    tmp20 = tmp10 * tmp19
    tmp21 = tmp0 - tmp3
    tmp22 = tmp11 - tmp14
    tmp23 = tmp21 * tmp22
    tmp24 = tmp1 - tmp4
    tmp25 = tmp12 - tmp15
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp20
    tmp29 = tmp3 + tmp0
    tmp30 = 2.0
    tmp31 = tmp29 / tmp30
    tmp32 = tmp4 + tmp1
    tmp33 = tmp32 / tmp30
    tmp34 = tmp31 - tmp33
    tmp35 = tmp34 * tmp34
    tmp36 = tmp14 + tmp11
    tmp37 = tmp36 / tmp30
    tmp38 = tmp15 + tmp12
    tmp39 = tmp38 / tmp30
    tmp40 = tmp37 - tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tmp35 + tmp41
    tmp43 = triton_helpers.minimum(tmp3, tmp4)
    tmp44 = triton_helpers.maximum(tmp0, tmp1)
    tmp45 = tmp43 - tmp44
    tmp46 = tmp45 * tmp45
    tmp47 = triton_helpers.minimum(tmp14, tmp15)
    tmp48 = triton_helpers.maximum(tmp11, tmp12)
    tmp49 = tmp47 - tmp48
    tmp50 = tmp49 * tmp49
    tmp51 = tmp46 + tmp50
    tmp52 = tmp51 + tmp7
    tmp53 = tmp42 / tmp52
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
    tl.store(out_ptr2 + (x0), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uu/cuuiepcd6ckdx4fcz7yzu3sybylu226ptneei5rpiupxs3sbxbt7.py
# Topologically Sorted Source Nodes: [iou, diou, loss, loss_1], Original ATen: [aten.div, aten.sub, aten.rsub, aten.mean]
# Source node to ATen node mapping:
#   diou => sub_9
#   iou => div
#   loss => sub_10
#   loss_1 => mean_4
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, %sub_5), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %div_1), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sub_9), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_10,), kwargs = {})
triton_poi_fused_div_mean_rsub_sub_1 = async_compile.triton('triton_poi_fused_div_mean_rsub_sub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mean_rsub_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mean_rsub_sub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp10 = tl.load(in_ptr0 + (1))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.load(in_ptr1 + (1))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr2 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (2))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp22 = tl.load(in_ptr1 + (2))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp25 = tl.load(in_ptr2 + (2))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp30 = tl.load(in_ptr0 + (3))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp32 = tl.load(in_ptr1 + (3))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp35 = tl.load(in_ptr2 + (3))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp4 = tmp1 / tmp3
    tmp7 = tmp4 - tmp6
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp14 = tmp11 / tmp13
    tmp17 = tmp14 - tmp16
    tmp18 = tmp8 - tmp17
    tmp19 = tmp9 + tmp18
    tmp24 = tmp21 / tmp23
    tmp27 = tmp24 - tmp26
    tmp28 = tmp8 - tmp27
    tmp29 = tmp19 + tmp28
    tmp34 = tmp31 / tmp33
    tmp37 = tmp34 - tmp36
    tmp38 = tmp8 - tmp37
    tmp39 = tmp29 + tmp38
    tmp40 = 4.0
    tmp41 = tmp39 / tmp40
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp41, None)
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
        buf0 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf1 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub, sub_1, gt_area, sub_2, sub_3, pr_area, add_1, inter, union, gt_cent_x, pr_cent_x, sub_6, pow_1, gt_cent_y, pr_cent_y, sub_7, pow_2, cent_dis, lt_1, rb_1, sub_8, pow_3, diag_dis, add_3, reg], Original ATen: [aten.sub, aten.mul, aten.add, aten.mean, aten.pow, aten.minimum, aten.maximum, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_maximum_mean_minimum_mul_pow_sub_sum_0.run(arg0_1, arg1_1, buf0, buf1, buf2, 4, grid=grid(4), stream=stream0)
        del arg0_1
        del arg1_1
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [iou, diou, loss, loss_1], Original ATen: [aten.div, aten.sub, aten.rsub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_mean_rsub_sub_1.run(buf0, buf1, buf2, buf3, 1, grid=grid(1), stream=stream0)
        del buf0
        del buf1
        del buf2
    return (buf3, )


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
