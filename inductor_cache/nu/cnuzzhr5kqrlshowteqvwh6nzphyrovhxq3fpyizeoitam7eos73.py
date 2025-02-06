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


# kernel path: inductor_cache/yd/cydavr7o37sj42ljox7ft2nfqs2gmmpw3xwabakkxfs6jw6fw2l6.py
# Topologically Sorted Source Nodes: [align, pos_dist, log_sigmoid_1, neg_dist, neg, uniform, mul, add], Original ATen: [aten.log_sigmoid_forward, aten.div, aten.neg, aten.mean, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add => add
#   align => abs_1, exp, full_default, log1p, minimum, neg, sub
#   log_sigmoid_1 => abs_2, exp_1, full_default_1, log1p_1, minimum_1, neg_2, sub_1
#   mul => mul
#   neg => neg_1
#   neg_dist => div_1
#   pos_dist => div
#   uniform => mean
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_3, 1.0), kwargs = {})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %div), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%div,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_7, 1.0), kwargs = {})
#   %neg_1 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%div_1,), kwargs = {})
#   %minimum_1 : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_1, %neg_1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%neg_1,), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_2,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum_1, %log1p_1), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_1, [1]), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub, %mul), kwargs = {})
triton_poi_fused_add_div_log_sigmoid_forward_mean_mul_neg_0 = async_compile.triton('triton_poi_fused_add_div_log_sigmoid_forward_mean_mul_neg_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_log_sigmoid_forward_mean_mul_neg_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_log_sigmoid_forward_mean_mul_neg_0(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp10 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.minimum(tmp3, tmp2)
    tmp5 = tl_math.abs(tmp2)
    tmp6 = -tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = libdevice.log1p(tmp7)
    tmp9 = tmp4 - tmp8
    tmp11 = tmp10 * tmp1
    tmp12 = -tmp11
    tmp13 = triton_helpers.minimum(tmp3, tmp12)
    tmp14 = tl_math.abs(tmp12)
    tmp15 = -tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = libdevice.log1p(tmp16)
    tmp18 = tmp13 - tmp17
    tmp20 = tmp19 * tmp1
    tmp21 = -tmp20
    tmp22 = triton_helpers.minimum(tmp3, tmp21)
    tmp23 = tl_math.abs(tmp21)
    tmp24 = -tmp23
    tmp25 = tl_math.exp(tmp24)
    tmp26 = libdevice.log1p(tmp25)
    tmp27 = tmp22 - tmp26
    tmp28 = tmp18 + tmp27
    tmp30 = tmp29 * tmp1
    tmp31 = -tmp30
    tmp32 = triton_helpers.minimum(tmp3, tmp31)
    tmp33 = tl_math.abs(tmp31)
    tmp34 = -tmp33
    tmp35 = tl_math.exp(tmp34)
    tmp36 = libdevice.log1p(tmp35)
    tmp37 = tmp32 - tmp36
    tmp38 = tmp28 + tmp37
    tmp40 = tmp39 * tmp1
    tmp41 = -tmp40
    tmp42 = triton_helpers.minimum(tmp3, tmp41)
    tmp43 = tl_math.abs(tmp41)
    tmp44 = -tmp43
    tmp45 = tl_math.exp(tmp44)
    tmp46 = libdevice.log1p(tmp45)
    tmp47 = tmp42 - tmp46
    tmp48 = tmp38 + tmp47
    tmp49 = 4.0
    tmp50 = tmp48 / tmp49
    tmp51 = tmp50 * tmp1
    tmp52 = tmp9 + tmp51
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr0 + (x0), tmp50, xmask)
    tl.store(out_ptr1 + (x0), tmp52, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    assert_size_stride(arg2_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg1_1, (4, 1, 4), (4, 4, 1), 0), reinterpret_tensor(arg0_1, (4, 4, 1), (4, 1, 1), 0), out=buf0)
        del arg0_1
        buf2 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg1_1, (1, 4, 4), (16, 4, 1), 0), reinterpret_tensor(arg2_1, (1, 4, 4), (0, 1, 4), 0), out=buf2)
        del arg1_1
        del arg2_1
        buf1 = reinterpret_tensor(buf0, (4, ), (1, ), 0); del buf0  # reuse
        buf3 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf4 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [align, pos_dist, log_sigmoid_1, neg_dist, neg, uniform, mul, add], Original ATen: [aten.log_sigmoid_forward, aten.div, aten.neg, aten.mean, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_log_sigmoid_forward_mean_mul_neg_0.run(buf1, buf2, buf3, buf4, 4, grid=grid(4), stream=stream0)
        del buf2
    return (buf4, buf1, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
