# AOT ID: ['43_inference']
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


# kernel path: inductor_cache/lf/clfgul7wsownmtwkwzjkbsszj7vqokcuih7gv2uzgagazeqzm7kv.py
# Topologically Sorted Source Nodes: [cosine_similarity, cosine_similarity_1], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul]
# Source node to ATen node mapping:
#   cosine_similarity => clamp_min, clamp_min_1, div, div_1, mul, pow_1, pow_2, pow_3, pow_4, sum_1, sum_2
#   cosine_similarity_1 => clamp_min_2, clamp_min_3, div_2, div_3, mul_1, pow_5, pow_6, pow_7, pow_8, sum_4, sum_5
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-08), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg1_1, %clamp_min), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_4, 1e-08), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %clamp_min_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %div), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1], True), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_6, 1e-08), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg1_1, %clamp_min_2), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg2_1, 2), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1], True), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_5, 0.5), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_8, 1e-08), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg2_1, %clamp_min_3), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %div_2), kwargs = {})
triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_0 = async_compile.triton('triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (x3), xmask)
    tmp17 = tl.load(in_ptr1 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr1 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (x3), xmask)
    tmp33 = tl.load(in_ptr2 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr2 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr2 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr2 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-08
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tmp18 = tmp17 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = triton_helpers.maximum(tmp28, tmp13)
    tmp30 = tmp16 / tmp29
    tmp31 = tmp15 * tmp30
    tmp34 = tmp33 * tmp33
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 + tmp36
    tmp39 = tmp38 * tmp38
    tmp40 = tmp37 + tmp39
    tmp42 = tmp41 * tmp41
    tmp43 = tmp40 + tmp42
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp13)
    tmp46 = tmp32 / tmp45
    tmp47 = tmp15 * tmp46
    tl.store(out_ptr0 + (x3), tmp31, xmask)
    tl.store(out_ptr1 + (x3), tmp47, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n6/cn6q54lx3maxhzsamycpjxq3a4dquxddwdxvvvc5vxbow2w55nq2.py
# Topologically Sorted Source Nodes: [cosine_similarity, positive_distance, cosine_similarity_1, negative_distance, sub_2, add, full_like, losses, mean], Original ATen: [aten.sum, aten.rsub, aten.sub, aten.add, aten.full_like, aten.maximum, aten.mean]
# Source node to ATen node mapping:
#   add => add
#   cosine_similarity => sum_3
#   cosine_similarity_1 => sum_6
#   full_like => full_default
#   losses => maximum
#   mean => mean
#   negative_distance => sub_1
#   positive_distance => sub
#   sub_2 => sub_2
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sum_3), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sum_6), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %sub_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_2, 4), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%add, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%maximum,), kwargs = {})
triton_per_fused_add_full_like_maximum_mean_rsub_sub_sum_1 = async_compile.triton('triton_per_fused_add_full_like_maximum_mean_rsub_sub_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_full_like_maximum_mean_rsub_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_full_like_maximum_mean_rsub_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp12 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp7 - tmp15
    tmp17 = tmp8 - tmp16
    tmp18 = 4.0
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.sum(tmp22, 1)[:, None]
    tmp25 = 64.0
    tmp26 = tmp24 / tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp26, None)
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
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cosine_similarity, cosine_similarity_1], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_0.run(arg1_1, arg0_1, arg2_1, buf0, buf1, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [cosine_similarity, positive_distance, cosine_similarity_1, negative_distance, sub_2, add, full_like, losses, mean], Original ATen: [aten.sum, aten.rsub, aten.sub, aten.add, aten.full_like, aten.maximum, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_full_like_maximum_mean_rsub_sub_sum_1.run(buf3, buf0, buf1, 1, 64, grid=grid(1), stream=stream0)
        del buf0
        del buf1
    return (buf3, )


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
