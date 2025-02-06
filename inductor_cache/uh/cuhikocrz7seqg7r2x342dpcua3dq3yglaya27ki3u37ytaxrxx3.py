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


# kernel path: inductor_cache/lb/clbiq24do23ixoype4idtqqreepwh6aygbaxotuu3qbls2psiqwh.py
# Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%arg0_1, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_0 = async_compile.triton('triton_poi_fused__softmax_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2t/c2tjj7b4das5tmkny2e34oseqwqpvo7dpy5zdvzaj2szbyxlyvki.py
# Topologically Sorted Source Nodes: [softmax, prediction], Original ATen: [aten._softmax, aten.argmax]
# Source node to ATen node mapping:
#   prediction => argmax
#   softmax => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %argmax : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%div, -2), kwargs = {})
triton_poi_fused__softmax_argmax_1 = async_compile.triton('triton_poi_fused__softmax_argmax_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_argmax_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_argmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 16*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 16*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 16*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4 + x0 + 16*x1), xmask)
    tmp10 = tl.load(in_ptr0 + (4 + 16*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (5 + 16*x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (6 + 16*x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (7 + 16*x1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (8 + x0 + 16*x1), xmask)
    tmp34 = tl.load(in_ptr0 + (8 + 16*x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (9 + 16*x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (10 + 16*x1), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (11 + 16*x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr0 + (12 + x0 + 16*x1), xmask)
    tmp57 = tl.load(in_ptr0 + (12 + 16*x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr0 + (13 + 16*x1), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr0 + (14 + 16*x1), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr0 + (15 + 16*x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp9 / tmp16
    tmp18 = tmp8 > tmp17
    tmp19 = tmp8 == tmp17
    tmp20 = tmp8 != tmp8
    tmp21 = tmp17 != tmp17
    tmp22 = tmp20 > tmp21
    tmp23 = tmp18 | tmp22
    tmp24 = tmp20 & tmp21
    tmp25 = tmp19 | tmp24
    tmp26 = tl.full([1], 0, tl.int64)
    tmp27 = tl.full([1], 1, tl.int64)
    tmp28 = tmp26 < tmp27
    tmp29 = tmp25 & tmp28
    tmp30 = tmp23 | tmp29
    tmp31 = tl.where(tmp30, tmp8, tmp17)
    tmp32 = tl.where(tmp30, tmp26, tmp27)
    tmp36 = tmp34 + tmp35
    tmp38 = tmp36 + tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp33 / tmp40
    tmp42 = tmp31 > tmp41
    tmp43 = tmp31 == tmp41
    tmp44 = tmp31 != tmp31
    tmp45 = tmp41 != tmp41
    tmp46 = tmp44 > tmp45
    tmp47 = tmp42 | tmp46
    tmp48 = tmp44 & tmp45
    tmp49 = tmp43 | tmp48
    tmp50 = tl.full([1], 2, tl.int64)
    tmp51 = tmp32 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tmp47 | tmp52
    tmp54 = tl.where(tmp53, tmp31, tmp41)
    tmp55 = tl.where(tmp53, tmp32, tmp50)
    tmp59 = tmp57 + tmp58
    tmp61 = tmp59 + tmp60
    tmp63 = tmp61 + tmp62
    tmp64 = tmp56 / tmp63
    tmp65 = tmp54 > tmp64
    tmp66 = tmp54 == tmp64
    tmp67 = tmp54 != tmp54
    tmp68 = tmp64 != tmp64
    tmp69 = tmp67 > tmp68
    tmp70 = tmp65 | tmp69
    tmp71 = tmp67 & tmp68
    tmp72 = tmp66 | tmp71
    tmp73 = tl.full([1], 3, tl.int64)
    tmp74 = tmp55 < tmp73
    tmp75 = tmp72 & tmp74
    tmp76 = tmp70 | tmp75
    tmp77 = tl.where(tmp76, tmp54, tmp64)
    tmp78 = tl.where(tmp76, tmp55, tmp73)
    tl.store(out_ptr0 + (x2), tmp78, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/my/cmyhwbqznzdoy2b75iljmflbwo2frt6mmmbwq6b4p52kau4oeywh.py
# Topologically Sorted Source Nodes: [scores, sum_1, truediv], Original ATen: [aten.eq, aten.sum, aten.div]
# Source node to ATen node mapping:
#   scores => eq
#   sum_1 => sum_2
#   truediv => div_1
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%argmax, %arg1_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%eq,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, 256.0), kwargs = {})
triton_per_fused_div_eq_sum_2 = async_compile.triton('triton_per_fused_div_eq_sum_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_eq_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_eq_sum_2(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
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
    r0 = (rindex % 64)
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (r2), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 == tmp2
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 0.00390625
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp10, None)
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
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_0.run(arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [softmax, prediction], Original ATen: [aten._softmax, aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_argmax_1.run(buf0, buf1, 64, grid=grid(64), stream=stream0)
        del buf0
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [scores, sum_1, truediv], Original ATen: [aten.eq, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_eq_sum_2.run(buf1, arg1_1, buf3, 1, 256, grid=grid(1), stream=stream0)
        del arg1_1
        del buf1
    return (buf3, )


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
