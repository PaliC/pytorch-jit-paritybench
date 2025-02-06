# AOT ID: ['0_inference']
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


# kernel path: inductor_cache/es/cespviast7parjylvj2mzki6tpryv3kjpdt2bowqnby65uttgexi.py
# Topologically Sorted Source Nodes: [intersection, sum_1, sum_2, sum_3], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   intersection => mul_1
#   sum_1 => sum_1
#   sum_2 => sum_2
#   sum_3 => sum_3
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %view_1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view, [1]), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_1, [1]), kwargs = {})
triton_per_fused_mul_sum_0 = async_compile.triton('triton_per_fused_mul_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_sum_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/si/csimhq3eslkd2eumtyettfbhyt3wrje5hnj5tjw5xvuintqnalmj.py
# Topologically Sorted Source Nodes: [bce, mul_2, mul_1, add, add_1, add_2, dice, sum_4, truediv_1, dice_1, add_3], Original ATen: [aten.binary_cross_entropy_with_logits, aten.mul, aten.add, aten.div, aten.sum, aten.rsub]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   bce => abs_1, exp, full_default, log1p, mean, minimum, mul, neg, sub, sub_1, sub_2
#   dice => div
#   dice_1 => sub_3
#   mul_1 => mul_2
#   mul_2 => mul_3
#   sum_4 => sum_4
#   truediv_1 => div_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg0_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %arg1_1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %arg1_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg1_1,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %sub_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_2,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 0.5), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 2.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 1e-05), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, %sum_3), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, 1e-05), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add, %add_2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%div,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_4, 4), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %sub_3), kwargs = {})
triton_per_fused_add_binary_cross_entropy_with_logits_div_mul_rsub_sum_1 = async_compile.triton('triton_per_fused_add_binary_cross_entropy_with_logits_div_mul_rsub_sum_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_binary_cross_entropy_with_logits_div_mul_rsub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 14, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_binary_cross_entropy_with_logits_div_mul_rsub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr1 + (r0), None)
    tmp20 = tl.load(in_ptr2 + (0))
    tmp21 = tl.broadcast_to(tmp20, [1])
    tmp26 = tl.load(in_ptr3 + (0))
    tmp27 = tl.broadcast_to(tmp26, [1])
    tmp28 = tl.load(in_ptr4 + (0))
    tmp29 = tl.broadcast_to(tmp28, [1])
    tmp33 = tl.load(in_ptr2 + (1))
    tmp34 = tl.broadcast_to(tmp33, [1])
    tmp37 = tl.load(in_ptr3 + (1))
    tmp38 = tl.broadcast_to(tmp37, [1])
    tmp39 = tl.load(in_ptr4 + (1))
    tmp40 = tl.broadcast_to(tmp39, [1])
    tmp45 = tl.load(in_ptr2 + (2))
    tmp46 = tl.broadcast_to(tmp45, [1])
    tmp49 = tl.load(in_ptr3 + (2))
    tmp50 = tl.broadcast_to(tmp49, [1])
    tmp51 = tl.load(in_ptr4 + (2))
    tmp52 = tl.broadcast_to(tmp51, [1])
    tmp57 = tl.load(in_ptr2 + (3))
    tmp58 = tl.broadcast_to(tmp57, [1])
    tmp61 = tl.load(in_ptr3 + (3))
    tmp62 = tl.broadcast_to(tmp61, [1])
    tmp63 = tl.load(in_ptr4 + (3))
    tmp64 = tl.broadcast_to(tmp63, [1])
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.minimum(tmp5, tmp3)
    tmp7 = tl_math.abs(tmp3)
    tmp8 = -tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = libdevice.log1p(tmp9)
    tmp11 = tmp6 - tmp10
    tmp12 = tmp4 - tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 256.0
    tmp17 = tmp15 / tmp16
    tmp18 = 0.5
    tmp19 = tmp17 * tmp18
    tmp22 = 2.0
    tmp23 = tmp21 * tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp30 = tmp27 + tmp29
    tmp31 = tmp30 + tmp24
    tmp32 = tmp25 / tmp31
    tmp35 = tmp34 * tmp22
    tmp36 = tmp35 + tmp24
    tmp41 = tmp38 + tmp40
    tmp42 = tmp41 + tmp24
    tmp43 = tmp36 / tmp42
    tmp44 = tmp32 + tmp43
    tmp47 = tmp46 * tmp22
    tmp48 = tmp47 + tmp24
    tmp53 = tmp50 + tmp52
    tmp54 = tmp53 + tmp24
    tmp55 = tmp48 / tmp54
    tmp56 = tmp44 + tmp55
    tmp59 = tmp58 * tmp22
    tmp60 = tmp59 + tmp24
    tmp65 = tmp62 + tmp64
    tmp66 = tmp65 + tmp24
    tmp67 = tmp60 / tmp66
    tmp68 = tmp56 + tmp67
    tmp69 = 0.25
    tmp70 = tmp68 * tmp69
    tmp71 = tmp1 - tmp70
    tmp72 = tmp19 + tmp71
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp72, None)
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
        buf1 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf3 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [intersection, sum_1, sum_2, sum_3], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_0.run(arg1_1, arg0_1, buf1, buf2, buf3, 4, 64, grid=grid(4), stream=stream0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [bce, mul_2, mul_1, add, add_1, add_2, dice, sum_4, truediv_1, dice_1, add_3], Original ATen: [aten.binary_cross_entropy_with_logits, aten.mul, aten.add, aten.div, aten.sum, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_binary_cross_entropy_with_logits_div_mul_rsub_sum_1.run(buf4, arg0_1, arg1_1, buf1, buf2, buf3, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
        del buf1
        del buf2
        del buf3
    return (buf4, )


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
