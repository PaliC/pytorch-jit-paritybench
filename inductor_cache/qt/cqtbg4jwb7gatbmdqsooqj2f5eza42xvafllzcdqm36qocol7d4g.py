# AOT ID: ['3_inference']
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


# kernel path: inductor_cache/oy/coy54v5gczpdyyjcmc7glxfj4npjfpmmaqhytnymflg24ruwva5c.py
# Topologically Sorted Source Nodes: [max_1, input_1, sort, cumsum], Original ATen: [aten.max, aten.sub, aten.sort, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum => cumsum
#   input_1 => sub
#   max_1 => max_1
#   sort => sort
# Graph fragment:
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%arg0_1, -1, True), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %getitem), kwargs = {})
#   %sort : [num_users=1] = call_function[target=torch.ops.aten.sort.default](args = (%sub, -1, True), kwargs = {})
#   %cumsum : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%getitem_2, -1), kwargs = {})
triton_per_fused_cumsum_max_sort_sub_0 = async_compile.triton('triton_per_fused_cumsum_max_sort_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_max_sort_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cumsum_max_sort_sub_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = r1
    tmp10 = tmp9.to(tl.int16)
    tmp11 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13, tmp14, = triton_helpers.sort_with_index(tmp11, tmp12, None, 1, stable=False, descending=True)
    tmp15 = tmp13.to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp17, = tl.associative_scan((tmp16,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r1 + 4*x0), tmp8, xmask)
    tl.store(out_ptr1 + (r1 + 4*x0), tmp13, xmask)
    tl.store(out_ptr2 + (r1 + 4*x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/32/c32er55l4awta6p4zb6eti24yislbwqvv5hklo76gecpimmtb3dd.py
# Topologically Sorted Source Nodes: [mul, input_cumsum, support, sum_1], Original ATen: [aten.mul, aten.sub, aten.gt, aten.sum]
# Source node to ATen node mapping:
#   input_cumsum => sub_1
#   mul => mul_1
#   sum_1 => sum_1
#   support => gt
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute, %getitem_2), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cumsum, 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Tensor](args = (%mul_1, %sub_1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%gt, [-1]), kwargs = {})
triton_poi_fused_gt_mul_sub_sum_1 = async_compile.triton('triton_poi_fused_gt_mul_sub_sum_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gt_mul_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gt_mul_sub_sum_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp1 * tmp0
    tmp4 = tmp3 - tmp1
    tmp5 = tmp2 > tmp4
    tmp6 = tmp5.to(tl.int64)
    tmp8 = 2.0
    tmp9 = tmp8 * tmp7
    tmp11 = tmp10 - tmp1
    tmp12 = tmp9 > tmp11
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tmp6 + tmp13
    tmp16 = 3.0
    tmp17 = tmp16 * tmp15
    tmp19 = tmp18 - tmp1
    tmp20 = tmp17 > tmp19
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tmp14 + tmp21
    tmp24 = 4.0
    tmp25 = tmp24 * tmp23
    tmp27 = tmp26 - tmp1
    tmp28 = tmp25 > tmp27
    tmp29 = tmp28.to(tl.int64)
    tmp30 = tmp22 + tmp29
    tl.store(out_ptr0 + (x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dd/cddeif337oxilaqonbl7ncna46canhse3yssbrrl5aegeytfktvc.py
# Topologically Sorted Source Nodes: [input_cumsum, sub_1, tau, tau_1, sub_2, output], Original ATen: [aten.sub, aten.gather, aten.div, aten.clamp]
# Source node to ATen node mapping:
#   input_cumsum => sub_1
#   output => clamp_min
#   sub_1 => sub_2
#   sub_2 => sub_3
#   tau => gather
#   tau_1 => div
# Graph fragment:
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cumsum, 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, 1), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_1, -1, %sub_2), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%gather, %unsqueeze), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %div), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %sub), kwargs = {})
triton_poi_fused_clamp_div_gather_sub_2 = async_compile.triton('triton_poi_fused_clamp_div_gather_sub_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_gather_sub_2', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_gather_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([XBLOCK], 4, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 4")
    tmp9 = tl.load(in_ptr2 + (tmp7 + 4*x1), xmask, eviction_policy='evict_last')
    tmp10 = 1.0
    tmp11 = tmp9 - tmp10
    tmp12 = tmp1.to(tl.float32)
    tmp13 = tmp11 / tmp12
    tmp14 = tmp0 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
    tl.store(out_ptr1 + (x2), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_1, input_1, sort, cumsum], Original ATen: [aten.max, aten.sub, aten.sort, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_max_sort_sub_0.run(arg0_1, buf0, buf1, buf3, 64, 4, grid=grid(64), stream=stream0)
        buf4 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [mul, input_cumsum, support, sum_1], Original ATen: [aten.mul, aten.sub, aten.gt, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gt_mul_sub_sum_1.run(buf1, buf3, buf4, 64, grid=grid(64), stream=stream0)
        buf5 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_cumsum, sub_1, tau, tau_1, sub_2, output], Original ATen: [aten.sub, aten.gather, aten.div, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_div_gather_sub_2.run(buf0, buf4, buf3, buf5, arg0_1, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del buf0
        del buf3
        del buf4
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
