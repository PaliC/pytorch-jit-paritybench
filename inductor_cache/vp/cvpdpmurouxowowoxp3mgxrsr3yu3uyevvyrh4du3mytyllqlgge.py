# AOT ID: ['71_inference']
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


# kernel path: inductor_cache/3m/c3mlzhgdvhdudicorgvlawhhgb25hnvywdhq2qbxgh5whrz5z25u.py
# Topologically Sorted Source Nodes: [log_prob], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_prob => amax, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%arg0_1, [1], True), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %amax), kwargs = {})
triton_poi_fused__log_softmax_0 = async_compile.triton('triton_poi_fused__log_softmax_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lm/clmjkdmctd3t3vuhupffhq2tv6ee7vwiseew6kayylp6n5u67hgm.py
# Topologically Sorted Source Nodes: [log_prob], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_prob => exp, log, sub_1, sum_1
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %log), kwargs = {})
triton_poi_fused__log_softmax_1 = async_compile.triton('triton_poi_fused__log_softmax_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl_math.exp(tmp1)
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp2 + tmp4
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp5 + tmp7
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = tl_math.log(tmp11)
    tmp13 = tmp0 - tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ic/cic57wcncxbkdsuzlsa5nhatddefx6t3nuub3tgkbns3ciwzzmt3.py
# Topologically Sorted Source Nodes: [nll_loss_1, pad_mask, target, gather, nll_loss, nll_loss_2, mul, smooth_loss_1, sum_1, smooth_loss, smooth_loss_2, mul_1, loss], Original ATen: [aten.masked_fill, aten.eq, aten.gather, aten.neg, aten.mean, aten.mul, aten.sum, aten.add]
# Source node to ATen node mapping:
#   gather => gather
#   loss => add
#   mul => mul
#   mul_1 => mul_1
#   nll_loss => neg
#   nll_loss_1 => full_default_1, where_1
#   nll_loss_2 => mean
#   pad_mask => eq
#   smooth_loss => neg_1
#   smooth_loss_1 => full_default_2, where_2
#   smooth_loss_2 => mean_1
#   sum_1 => sum_2
#   target => full_default, where
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : [num_users=3] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg1_1, -100), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %arg1_1), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_1, 1, %where), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%gather,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_1, %neg), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1.0), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sub_1, [1], True), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sum_2,), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_2, %neg_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_1, 0.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
triton_per_fused_add_eq_gather_masked_fill_mean_mul_neg_sum_2 = async_compile.triton('triton_per_fused_add_eq_gather_masked_fill_mean_mul_neg_sum_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_eq_gather_masked_fill_mean_mul_neg_sum_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_eq_gather_masked_fill_mean_mul_neg_sum_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    r1 = rindex // 4
    tmp0 = tl.load(in_ptr0 + (r2), None)
    tmp17 = tl.load(in_ptr1 + (4*r1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (1 + 4*r1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (2 + 4*r1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr1 + (3 + 4*r1), None, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp5 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp4 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp4)
    tl.device_assert((0 <= tmp8) & (tmp8 < 4), "index out of bounds: 0 <= tmp8 < 4")
    tmp10 = tl.load(in_ptr1 + (tmp8 + 4*r1), None, eviction_policy='evict_last')
    tmp11 = -tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp12, tmp11)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = -tmp23
    tmp25 = tl.where(tmp2, tmp12, tmp24)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.sum(tmp26, 1)[:, None]
    tmp29 = 16.0
    tmp30 = tmp16 / tmp29
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp28 / tmp29
    tmp34 = tmp33 * tmp12
    tmp35 = tmp32 + tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp35, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_prob], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_0.run(arg0_1, buf0, 16, grid=grid(16), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_prob], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_1.run(buf0, buf1, 16, grid=grid(16), stream=stream0)
        del buf0
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [nll_loss_1, pad_mask, target, gather, nll_loss, nll_loss_2, mul, smooth_loss_1, sum_1, smooth_loss, smooth_loss_2, mul_1, loss], Original ATen: [aten.masked_fill, aten.eq, aten.gather, aten.neg, aten.mean, aten.mul, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_eq_gather_masked_fill_mean_mul_neg_sum_2.run(buf4, arg1_1, buf1, 1, 16, grid=grid(1), stream=stream0)
        del arg1_1
        del buf1
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
