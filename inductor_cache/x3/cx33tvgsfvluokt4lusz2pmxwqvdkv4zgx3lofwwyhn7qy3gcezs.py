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


# kernel path: inductor_cache/oc/coc2dbndgh3fv7tlnxz5riglmjoysbfv5npyeyt5gjtysoe7nxaf.py
# Topologically Sorted Source Nodes: [stack, lt, mul_8, mul_9, truediv_6, sub_12, loss, loss_1, loss_2], Original ATen: [aten.stack, aten.lt, aten.mul, aten.div, aten.sub, aten.where, aten.mean]
# Source node to ATen node mapping:
#   loss => where
#   loss_1 => mean
#   loss_2 => mul_10
#   lt => lt
#   mul_8 => mul_8
#   mul_9 => mul_9
#   stack => cat
#   sub_12 => sub_12
#   truediv_6 => div_6
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2, %unsqueeze_3], -1), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%view, 0.2), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, 0.5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %view), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_9, 0.2), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, 0.1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt, %div_6, %sub_12), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1.0), kwargs = {})
triton_per_fused_div_lt_mean_mul_stack_sub_where_0 = async_compile.triton('triton_per_fused_div_lt_mean_mul_stack_sub_where_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_lt_mean_mul_stack_sub_where_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 16, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_lt_mean_mul_stack_sub_where_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
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
    r0 = (rindex % 4)
    r1 = ((rindex // 4) % 16)
    r2 = rindex // 64
    r3 = rindex
    tmp0 = r0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(32 + r1 + 64*r2, [RBLOCK])), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (tl.broadcast_to(r1 + 64*r2, [RBLOCK])), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tmp6 + tmp5
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tl.load(in_ptr1 + (tl.broadcast_to(r1 + 64*r2, [RBLOCK])), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (tl.broadcast_to(32 + r1 + 64*r2, [RBLOCK])), tmp4, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13 * tmp9
    tmp15 = tmp10 - tmp14
    tmp16 = tl_math.abs(tmp15)
    tmp17 = 2.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp7 - tmp18
    tmp20 = tmp7 + tmp18
    tmp21 = 0.001
    tmp22 = tmp20 + tmp21
    tmp23 = tmp19 / tmp22
    tmp24 = 0.0
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp26 = 1.0
    tmp27 = tmp26 - tmp25
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp4, tmp27, tmp28)
    tmp30 = tmp0 >= tmp3
    tmp31 = tl.full([1], 2, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr0 + (tl.broadcast_to(48 + r1 + 64*r2, [RBLOCK])), tmp33, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr0 + (tl.broadcast_to(16 + r1 + 64*r2, [RBLOCK])), tmp33, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 - tmp35
    tmp37 = tmp35 + tmp34
    tmp38 = 0.5
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr1 + (tl.broadcast_to(16 + r1 + 64*r2, [RBLOCK])), tmp33, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr1 + (tl.broadcast_to(48 + r1 + 64*r2, [RBLOCK])), tmp33, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42 * tmp38
    tmp44 = tmp39 - tmp43
    tmp45 = tl_math.abs(tmp44)
    tmp46 = 2.0
    tmp47 = tmp45 * tmp46
    tmp48 = tmp36 - tmp47
    tmp49 = tmp36 + tmp47
    tmp50 = 0.001
    tmp51 = tmp49 + tmp50
    tmp52 = tmp48 / tmp51
    tmp53 = 0.0
    tmp54 = triton_helpers.maximum(tmp52, tmp53)
    tmp55 = 1.0
    tmp56 = tmp55 - tmp54
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp33, tmp56, tmp57)
    tmp59 = tmp0 >= tmp31
    tmp60 = tl.full([1], 3, tl.int64)
    tmp61 = tmp0 < tmp60
    tmp62 = tmp59 & tmp61
    tmp63 = tl.load(in_ptr0 + (tl.broadcast_to(32 + r1 + 64*r2, [RBLOCK])), tmp62, eviction_policy='evict_last', other=0.0)
    tmp64 = tl.load(in_ptr0 + (tl.broadcast_to(r1 + 64*r2, [RBLOCK])), tmp62, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 - tmp64
    tmp66 = tl.load(in_ptr1 + (tl.broadcast_to(32 + r1 + 64*r2, [RBLOCK])), tmp62, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.load(in_ptr1 + (tl.broadcast_to(r1 + 64*r2, [RBLOCK])), tmp62, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 - tmp67
    tmp69 = 0.001
    tmp70 = tmp68 + tmp69
    tmp71 = tmp65 / tmp70
    tmp72 = tmp65 + tmp69
    tmp73 = tmp68 / tmp72
    tmp74 = triton_helpers.minimum(tmp71, tmp73)
    tmp75 = 1.0
    tmp76 = tmp75 - tmp74
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp62, tmp76, tmp77)
    tmp79 = tmp0 >= tmp60
    tmp80 = tl.full([1], 4, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tl.load(in_ptr0 + (tl.broadcast_to(48 + r1 + 64*r2, [RBLOCK])), tmp79, eviction_policy='evict_last', other=0.0)
    tmp83 = tl.load(in_ptr0 + (tl.broadcast_to(16 + r1 + 64*r2, [RBLOCK])), tmp79, eviction_policy='evict_last', other=0.0)
    tmp84 = tmp82 - tmp83
    tmp85 = tl.load(in_ptr1 + (tl.broadcast_to(48 + r1 + 64*r2, [RBLOCK])), tmp79, eviction_policy='evict_last', other=0.0)
    tmp86 = tl.load(in_ptr1 + (tl.broadcast_to(16 + r1 + 64*r2, [RBLOCK])), tmp79, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 - tmp86
    tmp88 = 0.001
    tmp89 = tmp87 + tmp88
    tmp90 = tmp84 / tmp89
    tmp91 = tmp84 + tmp88
    tmp92 = tmp87 / tmp91
    tmp93 = triton_helpers.minimum(tmp90, tmp92)
    tmp94 = 1.0
    tmp95 = tmp94 - tmp93
    tmp96 = tl.full(tmp95.shape, 0.0, tmp95.dtype)
    tmp97 = tl.where(tmp79, tmp95, tmp96)
    tmp98 = tl.where(tmp62, tmp78, tmp97)
    tmp99 = tl.where(tmp33, tmp58, tmp98)
    tmp100 = tl.where(tmp4, tmp29, tmp99)
    tmp101 = 0.2
    tmp102 = tmp100 < tmp101
    tmp103 = 0.5
    tmp104 = tmp100 * tmp103
    tmp105 = tmp104 * tmp100
    tmp106 = 5.0
    tmp107 = tmp105 * tmp106
    tmp108 = 0.1
    tmp109 = tmp100 - tmp108
    tmp110 = tl.where(tmp102, tmp107, tmp109)
    tmp111 = tl.broadcast_to(tmp110, [RBLOCK])
    tmp113 = triton_helpers.promote_to_tensor(tl.sum(tmp111, 0))
    tmp114 = 256.0
    tmp115 = tmp113 / tmp114
    tmp116 = 1.0
    tmp117 = tmp115 * tmp116
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp117, None)
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
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [stack, lt, mul_8, mul_9, truediv_6, sub_12, loss, loss_1, loss_2], Original ATen: [aten.stack, aten.lt, aten.mul, aten.div, aten.sub, aten.where, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_lt_mean_mul_stack_sub_where_0.run(buf2, arg1_1, arg0_1, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf2, )


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
