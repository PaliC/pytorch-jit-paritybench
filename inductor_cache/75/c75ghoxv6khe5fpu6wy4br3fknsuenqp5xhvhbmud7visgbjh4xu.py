# AOT ID: ['5_inference']
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


# kernel path: inductor_cache/be/cbetowq3fqvitxqyvqyumlwplphxp2bke2a5x4igbtmvzp2rrmet.py
# Topologically Sorted Source Nodes: [binary_cross_entropy_with_logits, weight, pos_mask, sum_1, pos_num, truediv, setitem, neg_mask, sum_2, neg_num, truediv_1, mul, setitem_1, sum_3, weight_1], Original ATen: [aten.binary_cross_entropy_with_logits, aten.new_zeros, aten.eq, aten.sum, aten._to_copy, aten.reciprocal, aten.mul, aten.index_put, aten.div]
# Source node to ATen node mapping:
#   binary_cross_entropy_with_logits => abs_1, exp, full_default_1, log1p, minimum, mul_3, mul_4, neg, sub, sub_1, sub_2, sum_4
#   mul => mul_2
#   neg_mask => eq_1
#   neg_num => convert_element_type_1
#   pos_mask => eq
#   pos_num => convert_element_type
#   setitem => index_put
#   setitem_1 => index_put_1
#   sum_1 => sum_1
#   sum_2 => sum_2
#   sum_3 => sum_3
#   truediv => mul, reciprocal
#   truediv_1 => mul_1, reciprocal_1
#   weight => full_default
#   weight_1 => div
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg0_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %arg1_1), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_1, %arg1_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg1_1,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, %sub_1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : [num_users=2] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, 1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%eq,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_1, torch.float32), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%eq], %mul), kwargs = {})
#   %eq_1 : [num_users=2] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%eq_1,), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, 1.0), kwargs = {})
#   %index_put_1 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put, [%eq_1], %mul_2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%index_put_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%index_put_1, %sum_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %div), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_4,), kwargs = {})
triton_per_fused__to_copy_binary_cross_entropy_with_logits_div_eq_index_put_mul_new_zeros_reciprocal_sum_0 = async_compile.triton('triton_per_fused__to_copy_binary_cross_entropy_with_logits_div_eq_index_put_mul_new_zeros_reciprocal_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_binary_cross_entropy_with_logits_div_eq_index_put_mul_new_zeros_reciprocal_sum_0', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_binary_cross_entropy_with_logits_div_eq_index_put_mul_new_zeros_reciprocal_sum_0(in_out_ptr1, in_ptr0, in_ptr1, xnumel, rnumel):
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
    tmp27 = tl.load(in_ptr1 + (r0), None)
    tmp1 = 1.0
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = 0.0
    tmp8 = tmp0 == tmp7
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp6.to(tl.float32)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp14 / tmp13
    tmp16 = tmp15 * tmp1
    tmp17 = tl.where(tmp2, tmp16, tmp7)
    tmp18 = tmp12.to(tl.float32)
    tmp19 = tmp14 / tmp18
    tmp20 = tmp19 * tmp1
    tmp21 = tmp20 * tmp1
    tmp22 = tl.where(tmp8, tmp21, tmp17)
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tmp1 - tmp0
    tmp28 = tmp26 * tmp27
    tmp29 = triton_helpers.minimum(tmp7, tmp27)
    tmp30 = tl_math.abs(tmp27)
    tmp31 = -tmp30
    tmp32 = tl_math.exp(tmp31)
    tmp33 = libdevice.log1p(tmp32)
    tmp34 = tmp29 - tmp33
    tmp35 = tmp28 - tmp34
    tmp36 = tmp22 / tmp25
    tmp37 = tmp35 * tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tl.store(in_out_ptr1 + (tl.full([1], 0, tl.int32)), tmp40, None)
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
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [binary_cross_entropy_with_logits, weight, pos_mask, sum_1, pos_num, truediv, setitem, neg_mask, sum_2, neg_num, truediv_1, mul, setitem_1, sum_3, weight_1], Original ATen: [aten.binary_cross_entropy_with_logits, aten.new_zeros, aten.eq, aten.sum, aten._to_copy, aten.reciprocal, aten.mul, aten.index_put, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_binary_cross_entropy_with_logits_div_eq_index_put_mul_new_zeros_reciprocal_sum_0.run(buf5, arg0_1, arg1_1, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
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
