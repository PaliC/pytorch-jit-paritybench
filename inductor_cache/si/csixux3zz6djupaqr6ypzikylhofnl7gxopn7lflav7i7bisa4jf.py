# AOT ID: ['8_inference']
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


# kernel path: inductor_cache/fe/cfenylapoeaj4mzwcnh5xf6ek2j72uggw6lomly4lqmvlepngrpa.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, 2), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hf/chfjbdgavvzuk24upadgkxdngqw5sgqyq24ca7usdx64usgr74lq.py
# Topologically Sorted Source Nodes: [pow_1, sum_1, sub, pow_2, sum_2, dist], Original ATen: [aten.pow, aten.sum, aten.sub, aten.add]
# Source node to ATen node mapping:
#   dist => add
#   pow_1 => pow_1
#   pow_2 => pow_2
#   sub => sub
#   sum_1 => sum_1
#   sum_2 => sum_2
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_1, %mm), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [0], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub, %sum_2), kwargs = {})
triton_poi_fused_add_pow_sub_sum_1 = async_compile.triton('triton_poi_fused_add_pow_sub_sum_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_pow_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_pow_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (8 + x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (12 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = tmp13 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 + tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 + tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 + tmp22
    tmp24 = tmp12 + tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ov/cov2bhrmc6rio2hdai6wvxcf2zldhmndxppmjhgq3ohugir5t7lw.py
# Topologically Sorted Source Nodes: [neg, max_1], Original ATen: [aten.neg, aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
#   neg => neg
# Graph fragment:
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add,), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%neg, 1), kwargs = {})
triton_poi_fused_max_neg_2 = async_compile.triton('triton_poi_fused_max_neg_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_neg_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_neg_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp3 = -tmp2
    tmp4 = tmp1 > tmp3
    tmp5 = tmp1 == tmp3
    tmp6 = tmp1 != tmp1
    tmp7 = tmp3 != tmp3
    tmp8 = tmp6 > tmp7
    tmp9 = tmp4 | tmp8
    tmp10 = tmp6 & tmp7
    tmp11 = tmp5 | tmp10
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tl.full([1], 1, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp9 | tmp15
    tmp17 = tl.where(tmp16, tmp1, tmp3)
    tmp18 = tl.where(tmp16, tmp12, tmp13)
    tmp20 = -tmp19
    tmp21 = tmp17 > tmp20
    tmp22 = tmp17 == tmp20
    tmp23 = tmp17 != tmp17
    tmp24 = tmp20 != tmp20
    tmp25 = tmp23 > tmp24
    tmp26 = tmp21 | tmp25
    tmp27 = tmp23 & tmp24
    tmp28 = tmp22 | tmp27
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tmp18 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tmp26 | tmp31
    tmp33 = tl.where(tmp32, tmp17, tmp20)
    tmp34 = tl.where(tmp32, tmp18, tmp29)
    tmp36 = -tmp35
    tmp37 = tmp33 > tmp36
    tmp38 = tmp33 == tmp36
    tmp39 = tmp33 != tmp33
    tmp40 = tmp36 != tmp36
    tmp41 = tmp39 > tmp40
    tmp42 = tmp37 | tmp41
    tmp43 = tmp39 & tmp40
    tmp44 = tmp38 | tmp43
    tmp45 = tl.full([1], 3, tl.int64)
    tmp46 = tmp34 < tmp45
    tmp47 = tmp44 & tmp46
    tmp48 = tmp42 | tmp47
    tmp49 = tl.where(tmp48, tmp33, tmp36)
    tmp50 = tl.where(tmp48, tmp34, tmp45)
    tl.store(out_ptr0 + (x0), tmp50, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yn/cyns7jigxa6pguszge7lxyu3lfxthamzbmpiasuzlokerwgr5zyb.py
# Topologically Sorted Source Nodes: [quantize, sub_2, quantize_1, sub_1, pow_3, diff], Original ATen: [aten.embedding, aten.sub, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   diff => mean
#   pow_3 => pow_3
#   quantize => embedding
#   quantize_1 => add_1
#   sub_1 => sub_1
#   sub_2 => sub_2
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%permute, %view_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%embedding, %arg0_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %sub_2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%embedding, %arg0_1), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
triton_red_fused_add_embedding_mean_pow_sub_3 = async_compile.triton('triton_red_fused_add_embedding_mean_pow_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_pow_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_embedding_mean_pow_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        r1 = rindex // 4
        r0 = (rindex % 4)
        tmp0 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp5 < 4")
        tmp7 = tl.load(in_ptr2 + (tmp5 + 4*r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp7 - tmp0
        tmp9 = tmp0 + tmp8
        tmp10 = tmp8 * tmp8
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tl.store(out_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp9, rmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp14 = 256.0
    tmp15 = tmp12 / tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp15, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        buf1 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, matmul], Original ATen: [aten.mul, aten.mm]
        extern_kernels.mm(buf0, arg1_1, out=buf1)
        del buf0
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [pow_1, sum_1, sub, pow_2, sum_2, dist], Original ATen: [aten.pow, aten.sum, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_pow_sub_sum_1.run(buf2, arg0_1, arg1_1, 256, grid=grid(256), stream=stream0)
        buf3 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [neg, max_1], Original ATen: [aten.neg, aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_neg_2.run(buf2, buf3, 64, grid=grid(64), stream=stream0)
        buf4 = reinterpret_tensor(buf2, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf2  # reuse
        buf5 = empty_strided_cuda((), (), torch.float32)
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [quantize, sub_2, quantize_1, sub_1, pow_3, diff], Original ATen: [aten.embedding, aten.sub, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mean_pow_sub_3.run(buf6, arg0_1, buf3, arg1_1, buf4, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf4, buf6, reinterpret_tensor(buf3, (4, 4, 4), (16, 4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
