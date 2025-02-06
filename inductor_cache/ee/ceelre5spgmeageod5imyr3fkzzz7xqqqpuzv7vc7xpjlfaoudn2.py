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


# kernel path: inductor_cache/he/chef3sa75bhone2wx5yd62767bhw7zncjqxifhel6ebobbu6h7bi.py
# Topologically Sorted Source Nodes: [a, b], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   a => cat
#   b => cat_1
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos, %cos_1], 1), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos_2, %cos_3], 1), kwargs = {})
triton_poi_fused_stack_0 = async_compile.triton('triton_poi_fused_stack_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (12 + 8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (4 + 8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = -tmp6
    tmp8 = tmp5 + tmp7
    tmp9 = tl.load(in_ptr0 + (8 + 8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = -tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = libdevice.atan2(tmp8, tmp12)
    tmp14 = 1.5707963
    tmp15 = libdevice.fmod(tmp13, tmp14)
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tl_math.cos(tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 8, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr0 + (12 + 8*x1 + ((-4) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr0 + (4 + 8*x1 + ((-4) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = -tmp24
    tmp26 = tmp23 + tmp25
    tmp27 = tl.load(in_ptr0 + (8 + 8*x1 + ((-4) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr0 + (8*x1 + ((-4) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = -tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = libdevice.atan2(tmp26, tmp30)
    tmp32 = 1.5707963
    tmp33 = libdevice.fmod(tmp31, tmp32)
    tmp34 = tl_math.abs(tmp33)
    tmp35 = tmp34 - tmp32
    tmp36 = tl_math.cos(tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp20, tmp36, tmp37)
    tmp39 = tl.where(tmp4, tmp19, tmp38)
    tmp40 = tmp16 + tmp14
    tmp41 = tl_math.cos(tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp4, tmp41, tmp42)
    tmp44 = tl_math.cos(tmp34)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp20, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp43, tmp46)
    tl.store(out_ptr0 + (x2), tmp39, xmask)
    tl.store(out_ptr1 + (x2), tmp47, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ew/cewezklqzwshr46q4lohjfq44koawz56xvnmzlc77w7yu6wvk4a5.py
# Topologically Sorted Source Nodes: [R_tensor], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   R_tensor => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1], 1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 2)
    x0 = (xindex % 8)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 8*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 8*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/su/csu54hsq4of3xkmzf5bgj4v2dip2ww5ghltqtdsvlb2dyl3jt3ou.py
# Topologically Sorted Source Nodes: [rot_points], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   rot_points => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 2)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2 + 8*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tq/ctqofpgot63hd423cqcmc6xi6dey7kvzymuew6gb5l6eoupkgn4k.py
# Topologically Sorted Source Nodes: [sub_1, sub_2, areas, min_2], Original ATen: [aten.sub, aten.mul, aten.min]
# Source node to ATen node mapping:
#   areas => mul
#   min_2 => min_2
#   sub_1 => sub_1
#   sub_2 => sub_2
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_2, %select_3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %sub_2), kwargs = {})
#   %min_2 : [num_users=1] = call_function[target=torch.ops.aten.min.default](args = (%mul,), kwargs = {})
triton_per_fused_min_mul_sub_3 = async_compile.triton('triton_per_fused_min_mul_sub_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_min_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_min_mul_sub_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = (rindex % 4)
    r1 = rindex // 4
    tmp0 = tl.load(in_ptr0 + (r0 + 16*r1), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (4 + r0 + 16*r1), rmask, other=0.0)
    tmp5 = tl.load(in_ptr0 + (8 + r0 + 16*r1), rmask, other=0.0)
    tmp6 = tl.load(in_ptr0 + (12 + r0 + 16*r1), rmask, other=0.0)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = triton_helpers.minimum(tmp0, tmp1)
    tmp4 = tmp2 - tmp3
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = triton_helpers.minimum(tmp5, tmp6)
    tmp9 = tmp7 - tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, float("inf"))
    tmp14 = triton_helpers.min2(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp14, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 2, 4), (8, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, 8), (8, 1), torch.float32)
        buf1 = empty_strided_cuda((3, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [a, b], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_0.run(arg0_1, buf0, buf1, 24, grid=grid(24), stream=stream0)
        buf2 = empty_strided_cuda((3, 2, 2, 4), (16, 8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [R_tensor], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf0, buf1, buf2, 48, grid=grid(48), stream=stream0)
        del buf0
        del buf1
        buf3 = empty_strided_cuda((3, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rot_points], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(arg0_1, buf3, 96, grid=grid(96), stream=stream0)
        del arg0_1
        buf4 = empty_strided_cuda((6, 2, 4), (8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rot_points], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (6, 2, 4), (8, 4, 1), 0), reinterpret_tensor(buf3, (6, 4, 4), (16, 4, 1), 0), out=buf4)
        del buf2
        del buf3
        buf5 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, sub_2, areas, min_2], Original ATen: [aten.sub, aten.mul, aten.min]
        stream0 = get_raw_stream(0)
        triton_per_fused_min_mul_sub_3.run(buf4, buf5, 1, 12, grid=grid(1), stream=stream0)
        del buf4
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 2, 4), (8, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
