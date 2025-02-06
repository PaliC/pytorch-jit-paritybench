# AOT ID: ['10_inference']
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


# kernel path: inductor_cache/s4/cs4stkiqtltwutotwphrvjozd6qsgugsqfi7irbgxqpmka665vrk.py
# Topologically Sorted Source Nodes: [mul, sub, mul_1, alpha_weight, pred_sigmoid, sub_1, mul_2, sub_2, mul_3, pt, pow_1, focal_weight, clamp, mul_5, sub_3, abs_1, neg, exp, log1p, loss, loss_1, mul_7], Original ATen: [aten.mul, aten.rsub, aten.add, aten.sigmoid, aten.pow, aten.clamp, aten.sub, aten.abs, aten.neg, aten.exp, aten.log1p]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   alpha_weight => add
#   clamp => clamp_min
#   exp => exp
#   focal_weight => mul_4
#   log1p => log1p
#   loss => add_2
#   loss_1 => mul_6
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_5 => mul_5
#   mul_7 => mul_7
#   neg => neg
#   pow_1 => pow_1
#   pred_sigmoid => sigmoid
#   pt => add_1
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.25), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg1_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, 0.75), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg0_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %sub_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %arg1_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %sigmoid), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1, 2.0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %pow_1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg0_1, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %mul_5), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg0_1,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_3, %log1p), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_2), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %arg2_1), kwargs = {})
triton_poi_fused_abs_add_clamp_exp_log1p_mul_neg_pow_rsub_sigmoid_sub_0 = async_compile.triton('triton_poi_fused_abs_add_clamp_exp_log1p_mul_neg_pow_rsub_sigmoid_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_exp_log1p_mul_neg_pow_rsub_sigmoid_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_exp_log1p_mul_neg_pow_rsub_sigmoid_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = 1.0
    tmp4 = tmp3 - tmp0
    tmp5 = 0.75
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp3 - tmp9
    tmp11 = tmp0 * tmp10
    tmp12 = tmp4 * tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp8, tmp16)
    tmp18 = tmp8 * tmp0
    tmp19 = tmp17 - tmp18
    tmp20 = tl_math.abs(tmp8)
    tmp21 = -tmp20
    tmp22 = tl_math.exp(tmp21)
    tmp23 = libdevice.log1p(tmp22)
    tmp24 = tmp19 + tmp23
    tmp25 = tmp15 * tmp24
    tmp27 = tmp25 * tmp26
    tl.store(out_ptr0 + (x0), tmp27, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, sub, mul_1, alpha_weight, pred_sigmoid, sub_1, mul_2, sub_2, mul_3, pt, pow_1, focal_weight, clamp, mul_5, sub_3, abs_1, neg, exp, log1p, loss, loss_1, mul_7], Original ATen: [aten.mul, aten.rsub, aten.add, aten.sigmoid, aten.pow, aten.clamp, aten.sub, aten.abs, aten.neg, aten.exp, aten.log1p]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_clamp_exp_log1p_mul_neg_pow_rsub_sigmoid_sub_0.run(arg1_1, arg0_1, arg2_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
