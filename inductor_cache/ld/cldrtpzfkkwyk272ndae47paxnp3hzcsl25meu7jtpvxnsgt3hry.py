# AOT ID: ['26_inference']
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


# kernel path: inductor_cache/rn/crnq2nsnldl2sndpyqsdjtdubxdmcvb3vfriw7lns7qlxfcczsac.py
# Topologically Sorted Source Nodes: [clamp, x, avg_pool2d, pow_2], Original ATen: [aten.clamp, aten.pow, aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d => avg_pool2d
#   clamp => clamp_min
#   pow_2 => pow_2
#   x => pow_1
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg0_1, 1e-07), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min, 3.0), kwargs = {})
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [4, 4]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%avg_pool2d, 0.3333333333333333), kwargs = {})
triton_poi_fused_avg_pool2d_clamp_pow_0 = async_compile.triton('triton_poi_fused_avg_pool2d_clamp_pow_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_clamp_pow_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_clamp_pow_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp1 = 1e-07
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 * tmp2
    tmp6 = triton_helpers.maximum(tmp5, tmp1)
    tmp7 = tmp6 * tmp6
    tmp8 = tmp7 * tmp6
    tmp9 = tmp8 + tmp4
    tmp11 = triton_helpers.maximum(tmp10, tmp1)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 * tmp11
    tmp14 = tmp13 + tmp9
    tmp16 = triton_helpers.maximum(tmp15, tmp1)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp17 * tmp16
    tmp19 = tmp18 + tmp14
    tmp21 = triton_helpers.maximum(tmp20, tmp1)
    tmp22 = tmp21 * tmp21
    tmp23 = tmp22 * tmp21
    tmp24 = tmp23 + tmp19
    tmp26 = triton_helpers.maximum(tmp25, tmp1)
    tmp27 = tmp26 * tmp26
    tmp28 = tmp27 * tmp26
    tmp29 = tmp28 + tmp24
    tmp31 = triton_helpers.maximum(tmp30, tmp1)
    tmp32 = tmp31 * tmp31
    tmp33 = tmp32 * tmp31
    tmp34 = tmp33 + tmp29
    tmp36 = triton_helpers.maximum(tmp35, tmp1)
    tmp37 = tmp36 * tmp36
    tmp38 = tmp37 * tmp36
    tmp39 = tmp38 + tmp34
    tmp41 = triton_helpers.maximum(tmp40, tmp1)
    tmp42 = tmp41 * tmp41
    tmp43 = tmp42 * tmp41
    tmp44 = tmp43 + tmp39
    tmp46 = triton_helpers.maximum(tmp45, tmp1)
    tmp47 = tmp46 * tmp46
    tmp48 = tmp47 * tmp46
    tmp49 = tmp48 + tmp44
    tmp51 = triton_helpers.maximum(tmp50, tmp1)
    tmp52 = tmp51 * tmp51
    tmp53 = tmp52 * tmp51
    tmp54 = tmp53 + tmp49
    tmp56 = triton_helpers.maximum(tmp55, tmp1)
    tmp57 = tmp56 * tmp56
    tmp58 = tmp57 * tmp56
    tmp59 = tmp58 + tmp54
    tmp61 = triton_helpers.maximum(tmp60, tmp1)
    tmp62 = tmp61 * tmp61
    tmp63 = tmp62 * tmp61
    tmp64 = tmp63 + tmp59
    tmp66 = triton_helpers.maximum(tmp65, tmp1)
    tmp67 = tmp66 * tmp66
    tmp68 = tmp67 * tmp66
    tmp69 = tmp68 + tmp64
    tmp71 = triton_helpers.maximum(tmp70, tmp1)
    tmp72 = tmp71 * tmp71
    tmp73 = tmp72 * tmp71
    tmp74 = tmp73 + tmp69
    tmp76 = triton_helpers.maximum(tmp75, tmp1)
    tmp77 = tmp76 * tmp76
    tmp78 = tmp77 * tmp76
    tmp79 = tmp78 + tmp74
    tmp80 = 0.0625
    tmp81 = tmp79 * tmp80
    tmp82 = 0.3333333333333333
    tmp83 = libdevice.pow(tmp81, tmp82)
    tl.store(in_out_ptr0 + (x0), tmp83, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf1 = reinterpret_tensor(buf0, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [clamp, x, avg_pool2d, pow_2], Original ATen: [aten.clamp, aten.pow, aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_clamp_pow_0.run(buf1, arg0_1, 16, grid=grid(16), stream=stream0)
        del arg0_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
