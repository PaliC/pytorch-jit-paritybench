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


# kernel path: inductor_cache/f3/cf3ptqr6irifg4mbd6s4lzl6sohkbm25xaqjl6ezm2lit67cwwus.py
# Topologically Sorted Source Nodes: [mul, mean, setitem, mul_1, mean_1, setitem_1, mul_2, mean_2, setitem_2, mul_3, mean_3, setitem_3], Original ATen: [aten.mul, aten.mean, aten.copy]
# Source node to ATen node mapping:
#   mean => mean
#   mean_1 => mean_1
#   mean_2 => mean_2
#   mean_3 => mean_3
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   setitem => copy
#   setitem_1 => copy_1
#   setitem_2 => copy_2
#   setitem_3 => copy_3
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul, [1]), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %mean), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_12, %slice_16), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_1, [1]), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_22, %mean_1), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int, %copy_1, 2, 1, 9223372036854775807), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_31, %slice_35), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_2, [1]), kwargs = {})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_41, %mean_2), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_1, %copy_2, 2, 2, 9223372036854775807), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_50, %slice_54), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_3, [1]), kwargs = {})
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_60, %mean_3), kwargs = {})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_2, %copy_3, 2, 3, 9223372036854775807), kwargs = {})
triton_poi_fused_copy_mean_mul_0 = async_compile.triton('triton_poi_fused_copy_mean_mul_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_mean_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_mean_mul_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x4 = xindex
    x2 = (xindex % 4)
    x3 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp4 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp8 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp12 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = x2
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 >= tmp18
    tmp20 = tl.load(in_ptr0 + (x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp21 = tl.load(in_ptr1 + ((-1) + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr0 + (16 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (15 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = tl.load(in_ptr0 + (32 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp28 = tl.load(in_ptr1 + (31 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 + tmp29
    tmp31 = tl.load(in_ptr0 + (48 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp32 = tl.load(in_ptr1 + (47 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp33 = tmp31 * tmp32
    tmp34 = tmp30 + tmp33
    tmp35 = 4.0
    tmp36 = tmp34 / tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp19, tmp36, tmp37)
    tmp39 = tl.full([1], 1, tl.int32)
    tmp40 = tl.full([1], 0, tl.int32)
    tmp41 = tmp39 == tmp40
    tmp42 = 0.0
    tmp43 = tl.where(tmp41, tmp16, tmp42)
    tmp44 = tl.where(tmp19, tmp38, tmp43)
    tmp45 = tl.full([1], 2, tl.int64)
    tmp46 = tmp17 >= tmp45
    tmp47 = tl.load(in_ptr0 + (x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp48 = tl.load(in_ptr1 + ((-2) + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp49 = tmp47 * tmp48
    tmp50 = tl.load(in_ptr0 + (16 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp51 = tl.load(in_ptr1 + (14 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp52 = tmp50 * tmp51
    tmp53 = tmp49 + tmp52
    tmp54 = tl.load(in_ptr0 + (32 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp55 = tl.load(in_ptr1 + (30 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp56 = tmp54 * tmp55
    tmp57 = tmp53 + tmp56
    tmp58 = tl.load(in_ptr0 + (48 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp59 = tl.load(in_ptr1 + (46 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp60 = tmp58 * tmp59
    tmp61 = tmp57 + tmp60
    tmp62 = 4.0
    tmp63 = tmp61 / tmp62
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp46, tmp63, tmp64)
    tmp66 = tl.full([1], 2, tl.int32)
    tmp67 = tmp66 == tmp39
    tmp68 = tmp66 == tmp40
    tmp69 = tl.where(tmp68, tmp16, tmp42)
    tmp70 = tl.where(tmp67, tmp44, tmp69)
    tmp71 = tl.where(tmp46, tmp65, tmp70)
    tmp72 = tl.full([1], 3, tl.int64)
    tmp73 = tmp17 >= tmp72
    tmp74 = tl.load(in_ptr0 + (3 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tl.load(in_ptr1 + (4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp74 * tmp75
    tmp77 = tl.load(in_ptr0 + (19 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tl.load(in_ptr1 + (16 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp77 * tmp78
    tmp80 = tmp76 + tmp79
    tmp81 = tl.load(in_ptr0 + (35 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp82 = tl.load(in_ptr1 + (32 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp81 * tmp82
    tmp84 = tmp80 + tmp83
    tmp85 = tl.load(in_ptr0 + (51 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tl.load(in_ptr1 + (48 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 * tmp86
    tmp88 = tmp84 + tmp87
    tmp89 = 4.0
    tmp90 = tmp88 / tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp73, tmp90, tmp91)
    tmp93 = tl.full([1], 3, tl.int32)
    tmp94 = tmp93 == tmp66
    tmp95 = tmp93 == tmp39
    tmp96 = tmp93 == tmp40
    tmp97 = tl.where(tmp96, tmp16, tmp42)
    tmp98 = tl.where(tmp95, tmp44, tmp97)
    tmp99 = tl.where(tmp94, tmp71, tmp98)
    tmp100 = tl.where(tmp73, tmp92, tmp99)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp44, xmask)
    tl.store(out_ptr2 + (x4), tmp71, xmask)
    tl.store(out_ptr3 + (x4), tmp100, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bx/cbxt73qxbpifza5geu346xvhhsaopcmbsa4f62cg3jwnnttzewrv.py
# Topologically Sorted Source Nodes: [cost_volume, mul, mean, setitem], Original ATen: [aten.new_zeros, aten.mul, aten.mean, aten.copy]
# Source node to ATen node mapping:
#   cost_volume => full_default
#   mean => mean
#   mul => mul
#   setitem => copy
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul, [1]), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %mean), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %copy, 1, 0), kwargs = {})
#   %select_scatter_default_1 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %slice_scatter_default, 1, 1), kwargs = {})
#   %select_scatter_default_2 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %slice_scatter_default_1, 1, 2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_2, %slice_scatter_default_2, 1, 3), kwargs = {})
triton_poi_fused_copy_mean_mul_new_zeros_1 = async_compile.triton('triton_poi_fused_copy_mean_mul_new_zeros_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_mean_mul_new_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_mean_mul_new_zeros_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 4)
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 2, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = tmp0 == tmp10
    tmp13 = 0.0
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.where(tmp8, tmp9, tmp14)
    tmp16 = tl.where(tmp5, tmp6, tmp15)
    tmp17 = tl.where(tmp2, tmp3, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
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
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf1 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf2 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, mean, setitem, mul_1, mean_1, setitem_1, mul_2, mean_2, setitem_2, mul_3, mean_3, setitem_3], Original ATen: [aten.mul, aten.mean, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_mean_mul_0.run(arg0_1, arg1_1, buf0, buf1, buf2, buf3, 64, grid=grid(64), stream=stream0)
        del arg0_1
        del arg1_1
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cost_volume, mul, mean, setitem], Original ATen: [aten.new_zeros, aten.mul, aten.mean, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_mean_mul_new_zeros_1.run(buf3, buf2, buf1, buf0, buf4, 256, grid=grid(256), stream=stream0)
        del buf0
        del buf1
        del buf2
        del buf3
    return (buf4, )


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
