# AOT ID: ['2_inference']
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


# kernel path: inductor_cache/27/c27j7i7sfxbpbkakg75s6hl5f63g3llupzdszfmc56jhdnf7pavj.py
# Topologically Sorted Source Nodes: [sub, pow_1, sum_1, L2_diff, L2_all_pred], Original ATen: [aten.sub, aten.pow, aten.sum, aten.sqrt]
# Source node to ATen node mapping:
#   L2_all_pred => sum_2
#   L2_diff => sqrt
#   pow_1 => pow_1
#   sub => sub
#   sum_1 => sum_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [3]), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_1,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sqrt, [2]), kwargs = {})
triton_poi_fused_pow_sqrt_sub_sum_0 = async_compile.triton('triton_poi_fused_pow_sqrt_sub_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_pow_sqrt_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_pow_sqrt_sub_sum_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr1 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr1 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr1 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr1 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp76 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr1 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tmp22 = tmp20 - tmp21
    tmp23 = tmp22 * tmp22
    tmp26 = tmp24 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tmp23 + tmp27
    tmp31 = tmp29 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tmp28 + tmp32
    tmp36 = tmp34 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tmp33 + tmp37
    tmp39 = libdevice.sqrt(tmp38)
    tmp40 = tmp19 + tmp39
    tmp43 = tmp41 - tmp42
    tmp44 = tmp43 * tmp43
    tmp47 = tmp45 - tmp46
    tmp48 = tmp47 * tmp47
    tmp49 = tmp44 + tmp48
    tmp52 = tmp50 - tmp51
    tmp53 = tmp52 * tmp52
    tmp54 = tmp49 + tmp53
    tmp57 = tmp55 - tmp56
    tmp58 = tmp57 * tmp57
    tmp59 = tmp54 + tmp58
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp40 + tmp60
    tmp64 = tmp62 - tmp63
    tmp65 = tmp64 * tmp64
    tmp68 = tmp66 - tmp67
    tmp69 = tmp68 * tmp68
    tmp70 = tmp65 + tmp69
    tmp73 = tmp71 - tmp72
    tmp74 = tmp73 * tmp73
    tmp75 = tmp70 + tmp74
    tmp78 = tmp76 - tmp77
    tmp79 = tmp78 * tmp78
    tmp80 = tmp75 + tmp79
    tmp81 = libdevice.sqrt(tmp80)
    tmp82 = tmp61 + tmp81
    tl.store(out_ptr0 + (x0), tmp82, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/34/c346rqt4zvbg5xzbmcuspzaurlybyqnbvcprq7tmahiz5cblfrns.py
# Topologically Sorted Source Nodes: [L2_mean_pred, L2_mean_pred_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   L2_mean_pred => mean
#   L2_mean_pred_1 => mean_1
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_2, [1]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mean, [0]), kwargs = {})
triton_poi_fused_mean_1 = async_compile.triton('triton_poi_fused_mean_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (1))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (2))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (3))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp13 = tl.load(in_ptr0 + (4))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (5))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp18 = tl.load(in_ptr0 + (6))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (7))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp26 = tl.load(in_ptr0 + (8))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp28 = tl.load(in_ptr0 + (9))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (10))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp34 = tl.load(in_ptr0 + (11))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp39 = tl.load(in_ptr0 + (12))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp41 = tl.load(in_ptr0 + (13))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.load(in_ptr0 + (14))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp47 = tl.load(in_ptr0 + (15))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tmp7 = tmp4 + tmp6
    tmp10 = tmp7 + tmp9
    tmp11 = 4.0
    tmp12 = tmp10 / tmp11
    tmp17 = tmp14 + tmp16
    tmp20 = tmp17 + tmp19
    tmp23 = tmp20 + tmp22
    tmp24 = tmp23 / tmp11
    tmp25 = tmp12 + tmp24
    tmp30 = tmp27 + tmp29
    tmp33 = tmp30 + tmp32
    tmp36 = tmp33 + tmp35
    tmp37 = tmp36 / tmp11
    tmp38 = tmp25 + tmp37
    tmp43 = tmp40 + tmp42
    tmp46 = tmp43 + tmp45
    tmp49 = tmp46 + tmp48
    tmp50 = tmp49 / tmp11
    tmp51 = tmp38 + tmp50
    tmp52 = tmp51 / tmp11
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp52, None)
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
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, pow_1, sum_1, L2_diff, L2_all_pred], Original ATen: [aten.sub, aten.pow, aten.sum, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_sqrt_sub_sum_0.run(arg0_1, arg1_1, buf0, 16, grid=grid(16), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [L2_mean_pred, L2_mean_pred_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_1.run(buf0, buf1, 1, grid=grid(1), stream=stream0)
        del buf0
    return (buf1, )


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
