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


# kernel path: inductor_cache/yv/cyvg47ku6kmnx7uyd6ww3hswowuyvnn6ec57b2q4gnl7mu5fdda7.py
# Topologically Sorted Source Nodes: [clamp, pow_1, avg_pool2d, truediv, pow_2], Original ATen: [aten.clamp, aten.pow, aten.avg_pool2d, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   avg_pool2d => avg_pool2d
#   clamp => clamp_min
#   pow_1 => pow_1
#   pow_2 => pow_2
#   truediv => mul, reciprocal
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg1_1, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%clamp_min, %arg0_1), kwargs = {})
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [4, 4]), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%arg0_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%avg_pool2d, %mul), kwargs = {})
triton_poi_fused_avg_pool2d_clamp_mul_pow_reciprocal_0 = async_compile.triton('triton_poi_fused_avg_pool2d_clamp_mul_pow_reciprocal_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_clamp_mul_pow_reciprocal_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_clamp_mul_pow_reciprocal_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp1 = 1e-06
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp5 = libdevice.pow(tmp2, tmp4)
    tmp7 = triton_helpers.maximum(tmp6, tmp1)
    tmp8 = libdevice.pow(tmp7, tmp4)
    tmp9 = tmp8 + tmp5
    tmp11 = triton_helpers.maximum(tmp10, tmp1)
    tmp12 = libdevice.pow(tmp11, tmp4)
    tmp13 = tmp12 + tmp9
    tmp15 = triton_helpers.maximum(tmp14, tmp1)
    tmp16 = libdevice.pow(tmp15, tmp4)
    tmp17 = tmp16 + tmp13
    tmp19 = triton_helpers.maximum(tmp18, tmp1)
    tmp20 = libdevice.pow(tmp19, tmp4)
    tmp21 = tmp20 + tmp17
    tmp23 = triton_helpers.maximum(tmp22, tmp1)
    tmp24 = libdevice.pow(tmp23, tmp4)
    tmp25 = tmp24 + tmp21
    tmp27 = triton_helpers.maximum(tmp26, tmp1)
    tmp28 = libdevice.pow(tmp27, tmp4)
    tmp29 = tmp28 + tmp25
    tmp31 = triton_helpers.maximum(tmp30, tmp1)
    tmp32 = libdevice.pow(tmp31, tmp4)
    tmp33 = tmp32 + tmp29
    tmp35 = triton_helpers.maximum(tmp34, tmp1)
    tmp36 = libdevice.pow(tmp35, tmp4)
    tmp37 = tmp36 + tmp33
    tmp39 = triton_helpers.maximum(tmp38, tmp1)
    tmp40 = libdevice.pow(tmp39, tmp4)
    tmp41 = tmp40 + tmp37
    tmp43 = triton_helpers.maximum(tmp42, tmp1)
    tmp44 = libdevice.pow(tmp43, tmp4)
    tmp45 = tmp44 + tmp41
    tmp47 = triton_helpers.maximum(tmp46, tmp1)
    tmp48 = libdevice.pow(tmp47, tmp4)
    tmp49 = tmp48 + tmp45
    tmp51 = triton_helpers.maximum(tmp50, tmp1)
    tmp52 = libdevice.pow(tmp51, tmp4)
    tmp53 = tmp52 + tmp49
    tmp55 = triton_helpers.maximum(tmp54, tmp1)
    tmp56 = libdevice.pow(tmp55, tmp4)
    tmp57 = tmp56 + tmp53
    tmp59 = triton_helpers.maximum(tmp58, tmp1)
    tmp60 = libdevice.pow(tmp59, tmp4)
    tmp61 = tmp60 + tmp57
    tmp63 = triton_helpers.maximum(tmp62, tmp1)
    tmp64 = libdevice.pow(tmp63, tmp4)
    tmp65 = tmp64 + tmp61
    tmp66 = 0.0625
    tmp67 = tmp65 * tmp66
    tmp68 = tl.full([1], 1, tl.int32)
    tmp69 = tmp68 / tmp4
    tmp70 = 1.0
    tmp71 = tmp69 * tmp70
    tmp72 = libdevice.pow(tmp67, tmp71)
    tl.store(in_out_ptr0 + (x0), tmp72, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, ), (1, ))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf1 = reinterpret_tensor(buf0, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [clamp, pow_1, avg_pool2d, truediv, pow_2], Original ATen: [aten.clamp, aten.pow, aten.avg_pool2d, aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_clamp_mul_pow_reciprocal_0.run(buf1, arg1_1, arg0_1, 16, grid=grid(16), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
