# AOT ID: ['64_inference']
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


# kernel path: inductor_cache/gl/cgllenxjy6zyejnutee7ccicmjyes5bjhk35vvvahwejntlpzi2t.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   x => div
# Graph fragment:
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %expand), kwargs = {})
triton_poi_fused_div_0 = async_compile.triton('triton_poi_fused_div_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-12
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/34/c34vuv7mfuvsyntfjq2tmhuoupad5zofrt6vlkmsbxojxasnr5u7.py
# Topologically Sorted Source Nodes: [sub_1, add_1, triplet_loss_value, and_, distinct_indices, yi_not_equal_yk, valid_labels, mask, and__4, add, k_equal, k_less_or_equal, mask_1, mask_2, mask_3, triplet_loss_value_1, sum_2, gt, sum_1, num_positive_triplets, add_2, triplet_loss_value_2], Original ATen: [aten.sub, aten.add, aten.relu, aten.bitwise_and, aten.bitwise_not, aten.eq, aten.bitwise_or, aten._to_copy, aten.mul, aten.sum, aten.gt, aten.div]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   and_ => bitwise_and
#   and__4 => bitwise_and_4
#   distinct_indices => bitwise_and_1
#   gt => gt
#   k_equal => eq_2
#   k_less_or_equal => bitwise_or
#   mask => bitwise_and_3
#   mask_1 => bitwise_and_5
#   mask_2 => bitwise_and_6
#   mask_3 => convert_element_type_1
#   num_positive_triplets => convert_element_type_2
#   sub_1 => sub_1
#   sum_1 => sum_2
#   sum_2 => sum_3
#   triplet_loss_value => relu
#   triplet_loss_value_1 => mul
#   triplet_loss_value_2 => div_1
#   valid_labels => bitwise_and_2
#   yi_not_equal_yk => bitwise_not_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_18, %unsqueeze_19), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_1, 0.3), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%unsqueeze_1, %unsqueeze_2), kwargs = {})
#   %bitwise_and_1 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %unsqueeze_3), kwargs = {})
#   %bitwise_not_1 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze_7,), kwargs = {})
#   %bitwise_and_2 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%unsqueeze_6, %bitwise_not_1), kwargs = {})
#   %bitwise_and_3 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_1, %bitwise_and_2), kwargs = {})
#   %bitwise_and_4 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%unsqueeze_9, %unsqueeze_11), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_15, -1), kwargs = {})
#   %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%add, %unsqueeze_17), kwargs = {})
#   %bitwise_or : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%eq_2, %unsqueeze_13), kwargs = {})
#   %bitwise_and_5 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_4, %bitwise_or), kwargs = {})
#   %bitwise_and_6 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_3, %bitwise_and_5), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_6, torch.float32), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %convert_element_type_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul, 1e-08), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%gt,), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1e-08), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %add_2), kwargs = {})
triton_per_fused__to_copy_add_bitwise_and_bitwise_not_bitwise_or_div_eq_gt_mul_relu_sub_sum_1 = async_compile.triton('triton_per_fused__to_copy_add_bitwise_and_bitwise_not_bitwise_or_div_eq_gt_mul_relu_sub_sum_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bitwise_and_bitwise_not_bitwise_or_div_eq_gt_mul_relu_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_bitwise_and_bitwise_not_bitwise_or_div_eq_gt_mul_relu_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
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
    r2 = ((rindex // 16) % 4)
    r1 = ((rindex // 4) % 4)
    r0 = (rindex % 4)
    r3 = rindex // 64
    r6 = (rindex % 16)
    r7 = rindex
    r4 = ((rindex // 4) % 16)
    tmp19 = tl.load(in_ptr0 + (r0 + 4*r2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (r0 + 4*r3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (r6), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr1 + (r4), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (r0 + 4*r2), None, eviction_policy='evict_last')
    tmp0 = r2
    tmp1 = r1
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (tmp5 != 0)
    tmp7 = tmp6 == 0
    tmp8 = r0
    tmp9 = tmp0 == tmp8
    tmp10 = tl.where(tmp9, tmp3, tmp4)
    tmp11 = (tmp10 != 0)
    tmp12 = tmp11 == 0
    tmp13 = tmp7 & tmp12
    tmp14 = tmp1 == tmp8
    tmp15 = tl.where(tmp14, tmp3, tmp4)
    tmp16 = (tmp15 != 0)
    tmp17 = tmp16 == 0
    tmp18 = tmp13 & tmp17
    tmp21 = tmp19 == tmp20
    tmp23 = tmp22 == tmp20
    tmp24 = tmp23 == 0
    tmp25 = tmp21 & tmp24
    tmp26 = tmp18 & tmp25
    tmp27 = -1.0
    tmp28 = tmp20 >= tmp27
    tmp29 = tmp28 == 0
    tmp30 = tmp19 >= tmp27
    tmp31 = tmp30 == 0
    tmp32 = tmp29 & tmp31
    tmp33 = tmp20 + tmp27
    tmp34 = tmp33 == tmp22
    tmp35 = tmp22 >= tmp27
    tmp36 = tmp35 == 0
    tmp37 = tmp34 | tmp36
    tmp38 = tmp32 & tmp37
    tmp39 = tmp26 & tmp38
    tmp41 = tmp3 - tmp40
    tmp43 = tmp3 - tmp42
    tmp44 = tmp41 - tmp43
    tmp45 = 0.3
    tmp46 = tmp44 + tmp45
    tmp47 = tl.full([1], 0, tl.int32)
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tmp49 = tmp39.to(tl.float32)
    tmp50 = tmp48 * tmp49
    tmp51 = tl.broadcast_to(tmp50, [RBLOCK])
    tmp53 = triton_helpers.promote_to_tensor(tl.sum(tmp51, 0))
    tmp54 = 1e-08
    tmp55 = tmp50 > tmp54
    tmp56 = tmp55.to(tl.int64)
    tmp57 = tl.broadcast_to(tmp56, [RBLOCK])
    tmp59 = triton_helpers.promote_to_tensor(tl.sum(tmp57, 0))
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60 + tmp54
    tmp62 = tmp53 / tmp61
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp62, None)
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
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_0.run(arg0_1, buf0, 16, grid=grid(16), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(buf0, (4, 4), (1, 4), 0), out=buf1)
        del buf0
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [sub_1, add_1, triplet_loss_value, and_, distinct_indices, yi_not_equal_yk, valid_labels, mask, and__4, add, k_equal, k_less_or_equal, mask_1, mask_2, mask_3, triplet_loss_value_1, sum_2, gt, sum_1, num_positive_triplets, add_2, triplet_loss_value_2], Original ATen: [aten.sub, aten.add, aten.relu, aten.bitwise_and, aten.bitwise_not, aten.eq, aten.bitwise_or, aten._to_copy, aten.mul, aten.sum, aten.gt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_bitwise_and_bitwise_not_bitwise_or_div_eq_gt_mul_relu_sub_sum_1.run(buf5, arg1_1, buf1, 1, 256, grid=grid(1), stream=stream0)
        del arg1_1
        del buf1
    return (buf5, )


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
