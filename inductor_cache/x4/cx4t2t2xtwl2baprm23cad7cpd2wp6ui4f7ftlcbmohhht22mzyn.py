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


# kernel path: inductor_cache/bs/cbs4olfhpi26i4l76dxutb4vo57bbeifwdclug2ipo3v4oxdbvid.py
# Topologically Sorted Source Nodes: [ge, labels, num_labels_pos, sub, num_labels_neg, num_total, w, ge_1, output_gt_zero, sub_1, mul, mul_1, mul_2, sub_2, exp, add_1, log, loss_val, mul_3, loss_pos_pix, loss_pos, mul_5, sub_5, sub_4, mul_4, loss_neg_pix, loss_neg, mul_6, final_loss], Original ATen: [aten.ge, aten._to_copy, aten.sum, aten.rsub, aten.add, aten.div, aten.sub, aten.mul, aten.exp, aten.log, aten.neg]
# Source node to ATen node mapping:
#   add_1 => add_1
#   exp => exp
#   final_loss => add_2
#   ge => ge
#   ge_1 => ge_1
#   labels => convert_element_type
#   log => log
#   loss_neg => sum_4
#   loss_neg_pix => neg_1
#   loss_pos => sum_3
#   loss_pos_pix => neg
#   loss_val => sub_3
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   num_labels_neg => sum_2
#   num_labels_pos => sum_1
#   num_total => add
#   output_gt_zero => convert_element_type_1
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_4 => sub_4
#   sub_5 => sub_5
#   w => div
# Graph fragment:
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg1_1, 0.5), kwargs = {})
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge, torch.float32), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%convert_element_type,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type), kwargs = {})
#   %sum_2 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%sub,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %sum_2), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, %add), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_1, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %sub_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %convert_element_type_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, 2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %mul_2), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_1,), kwargs = {})
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %log), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %sub_3), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_3,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%neg,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sum_3), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %sub_3), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_4,), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%neg_1,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %sum_4), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
triton_per_fused__to_copy_add_div_exp_ge_log_mul_neg_rsub_sub_sum_0 = async_compile.triton('triton_per_fused__to_copy_add_div_exp_ge_log_mul_neg_rsub_sub_sum_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_exp_ge_log_mul_neg_rsub_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_div_exp_ge_log_mul_neg_rsub_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
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
    tmp12 = tl.load(in_ptr1 + (r0), None)
    tmp1 = 0.5
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = 1.0
    tmp8 = tmp7 - tmp3
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = 0.0
    tmp14 = tmp12 >= tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp3 - tmp15
    tmp17 = tmp12 * tmp16
    tmp18 = tmp12 * tmp15
    tmp19 = 2.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 - tmp20
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp22 + tmp7
    tmp24 = tl_math.log(tmp23)
    tmp25 = tmp17 - tmp24
    tmp26 = tmp3 * tmp25
    tmp27 = -tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = tmp8 * tmp25
    tmp32 = -tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp36 = tmp6 + tmp11
    tmp37 = tmp11 / tmp36
    tmp38 = tmp37 * tmp30
    tmp39 = tmp7 - tmp37
    tmp40 = tmp39 * tmp35
    tmp41 = tmp38 + tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp41, None)
''', device_str='cuda')


cpp_fused_lift_fresh_prod_1 = async_compile.cpp_pybinding(['int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(2);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(1);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<int64_t>(4);
                        auto tmp7 = tmp5 ? tmp6 : tmp6;
                        auto tmp8 = static_cast<int64_t>(3);
                        auto tmp9 = tmp1 < tmp8;
                        auto tmp10 = tmp9 ? tmp6 : tmp6;
                        auto tmp11 = tmp3 ? tmp7 : tmp10;
                        tmp_acc0 = tmp_acc0 * tmp11;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [ge, labels, num_labels_pos, sub, num_labels_neg, num_total, w, ge_1, output_gt_zero, sub_1, mul, mul_1, mul_2, sub_2, exp, add_1, log, loss_val, mul_3, loss_pos_pix, loss_pos, mul_5, sub_5, sub_4, mul_4, loss_neg_pix, loss_neg, mul_6, final_loss], Original ATen: [aten.ge, aten._to_copy, aten.sum, aten.rsub, aten.add, aten.div, aten.sub, aten.mul, aten.exp, aten.log, aten.neg]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_div_exp_ge_log_mul_neg_rsub_sub_sum_0.run(buf5, arg1_1, arg0_1, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
    buf4 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_1(buf4)
    return (buf5, buf4, )


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
