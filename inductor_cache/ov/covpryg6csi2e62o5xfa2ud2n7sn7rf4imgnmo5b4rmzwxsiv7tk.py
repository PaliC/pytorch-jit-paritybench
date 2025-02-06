# AOT ID: ['15_inference']
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


# kernel path: inductor_cache/pi/cpiwdd6zrqmxqerckeulwjkf3yh3gevvvkmcwlkoxstfp4geo43f.py
# Topologically Sorted Source Nodes: [cosine, eye, no_diag, mul, pow_1, sum_1], Original ATen: [aten.mul, aten.eye, aten.rsub, aten.pow, aten.sum]
# Source node to ATen node mapping:
#   cosine => mul
#   eye => eq, full_default, full_default_1, iota_1, where
#   mul => mul_1
#   no_diag => sub
#   pow_1 => pow_1
#   sum_1 => sum_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute, %permute_1), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %iota_1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %full_default_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %where), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sub), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
triton_poi_fused_eye_mul_pow_rsub_sum_0 = async_compile.triton('triton_poi_fused_eye_mul_pow_rsub_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eye_mul_pow_rsub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eye_mul_pow_rsub_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp12 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4 + x0), xmask)
    tmp22 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (8 + x0), xmask)
    tmp32 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (12 + x0), xmask)
    tmp2 = tmp0 * tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 == tmp4
    tmp6 = 1.0
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp6 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tmp10 * tmp10
    tmp14 = tmp12 * tmp13
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = tmp3 == tmp15
    tmp17 = tl.where(tmp16, tmp6, tmp7)
    tmp18 = tmp6 - tmp17
    tmp19 = tmp14 * tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tmp11 + tmp20
    tmp24 = tmp22 * tmp23
    tmp25 = tl.full([1], 2, tl.int64)
    tmp26 = tmp3 == tmp25
    tmp27 = tl.where(tmp26, tmp6, tmp7)
    tmp28 = tmp6 - tmp27
    tmp29 = tmp24 * tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tmp21 + tmp30
    tmp34 = tmp32 * tmp33
    tmp35 = tl.full([1], 3, tl.int64)
    tmp36 = tmp3 == tmp35
    tmp37 = tl.where(tmp36, tmp6, tmp7)
    tmp38 = tmp6 - tmp37
    tmp39 = tmp34 * tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tmp31 + tmp40
    tl.store(out_ptr0 + (x0), tmp41, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4e/c4eif6puzcyo36odu6kg6objxdfbxsjzt22gbnlys5fu4ogg4u4b.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_1,), kwargs = {})
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp4 = tmp1 + tmp3
    tmp7 = tmp4 + tmp6
    tmp10 = tmp7 + tmp9
    tmp11 = 4.0
    tmp12 = tmp10 / tmp11
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [cosine, eye, no_diag, mul, pow_1, sum_1], Original ATen: [aten.mul, aten.eye, aten.rsub, aten.pow, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_eye_mul_pow_rsub_sum_0.run(arg0_1, buf0, 4, grid=grid(4), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_1.run(buf0, buf1, 1, grid=grid(1), stream=stream0)
        del buf0
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
