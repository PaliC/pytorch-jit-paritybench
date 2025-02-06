# AOT ID: ['1_inference']
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


# kernel path: inductor_cache/ed/ced5jguse5qlwp42sksu4c4izxu2lhzicqkrvqnbbqtxvyybingn.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.avg_pool2d]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => avg_pool2d
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [1, 0, 1, 0], 0.0), kwargs = {})
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%constant_pad_nd, [3, 3], [2, 2], [1, 1], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_constant_pad_nd_0 = async_compile.triton('triton_poi_fused_avg_pool2d_constant_pad_nd_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 3)
    x0 = (xindex % 3)
    x2 = xindex // 9
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 5, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = (-2) + 2*x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = (-2) + 2*x0
    tmp15 = tmp14 >= tmp12
    tmp16 = tmp13 & tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + ((-10) + 2*x0 + 8*x1 + 16*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp10, tmp18, tmp19)
    tmp21 = 2*x0
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = (-2) + 2*x1
    tmp27 = tl.full([1], 0, tl.int64)
    tmp28 = tmp26 >= tmp27
    tmp29 = (-1) + 2*x0
    tmp30 = tmp29 >= tmp27
    tmp31 = tmp28 & tmp30
    tmp32 = tmp31 & tmp25
    tmp33 = tl.load(in_ptr0 + ((-9) + 2*x0 + 8*x1 + 16*x2), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp25, tmp33, tmp34)
    tmp36 = tmp35 + tmp20
    tmp37 = 1 + 2*x0
    tmp38 = tmp37 >= tmp1
    tmp39 = tmp37 < tmp3
    tmp40 = tmp38 & tmp39
    tmp41 = tmp5 & tmp40
    tmp42 = (-2) + 2*x1
    tmp43 = tl.full([1], 0, tl.int64)
    tmp44 = tmp42 >= tmp43
    tmp45 = 2*x0
    tmp46 = tmp45 >= tmp43
    tmp47 = tmp44 & tmp46
    tmp48 = tmp47 & tmp41
    tmp49 = tl.load(in_ptr0 + ((-8) + 2*x0 + 8*x1 + 16*x2), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp41, tmp49, tmp50)
    tmp52 = tmp51 + tmp36
    tmp53 = 2*x1
    tmp54 = tmp53 >= tmp1
    tmp55 = tmp53 < tmp3
    tmp56 = tmp54 & tmp55
    tmp57 = tmp56 & tmp9
    tmp58 = (-1) + 2*x1
    tmp59 = tl.full([1], 0, tl.int64)
    tmp60 = tmp58 >= tmp59
    tmp61 = (-2) + 2*x0
    tmp62 = tmp61 >= tmp59
    tmp63 = tmp60 & tmp62
    tmp64 = tmp63 & tmp57
    tmp65 = tl.load(in_ptr0 + ((-6) + 2*x0 + 8*x1 + 16*x2), tmp64 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp57, tmp65, tmp66)
    tmp68 = tmp67 + tmp52
    tmp69 = tmp56 & tmp24
    tmp70 = (-1) + 2*x1
    tmp71 = tl.full([1], 0, tl.int64)
    tmp72 = tmp70 >= tmp71
    tmp73 = (-1) + 2*x0
    tmp74 = tmp73 >= tmp71
    tmp75 = tmp72 & tmp74
    tmp76 = tmp75 & tmp69
    tmp77 = tl.load(in_ptr0 + ((-5) + 2*x0 + 8*x1 + 16*x2), tmp76 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tl.full(tmp77.shape, 0.0, tmp77.dtype)
    tmp79 = tl.where(tmp69, tmp77, tmp78)
    tmp80 = tmp79 + tmp68
    tmp81 = tmp56 & tmp40
    tmp82 = (-1) + 2*x1
    tmp83 = tl.full([1], 0, tl.int64)
    tmp84 = tmp82 >= tmp83
    tmp85 = 2*x0
    tmp86 = tmp85 >= tmp83
    tmp87 = tmp84 & tmp86
    tmp88 = tmp87 & tmp81
    tmp89 = tl.load(in_ptr0 + ((-4) + 2*x0 + 8*x1 + 16*x2), tmp88 & xmask, eviction_policy='evict_last', other=0.0)
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp81, tmp89, tmp90)
    tmp92 = tmp91 + tmp80
    tmp93 = 1 + 2*x1
    tmp94 = tmp93 >= tmp1
    tmp95 = tmp93 < tmp3
    tmp96 = tmp94 & tmp95
    tmp97 = tmp96 & tmp9
    tmp98 = 2*x1
    tmp99 = tl.full([1], 0, tl.int64)
    tmp100 = tmp98 >= tmp99
    tmp101 = (-2) + 2*x0
    tmp102 = tmp101 >= tmp99
    tmp103 = tmp100 & tmp102
    tmp104 = tmp103 & tmp97
    tmp105 = tl.load(in_ptr0 + ((-2) + 2*x0 + 8*x1 + 16*x2), tmp104 & xmask, eviction_policy='evict_last', other=0.0)
    tmp106 = tl.full(tmp105.shape, 0.0, tmp105.dtype)
    tmp107 = tl.where(tmp97, tmp105, tmp106)
    tmp108 = tmp107 + tmp92
    tmp109 = tmp96 & tmp24
    tmp110 = 2*x1
    tmp111 = tl.full([1], 0, tl.int64)
    tmp112 = tmp110 >= tmp111
    tmp113 = (-1) + 2*x0
    tmp114 = tmp113 >= tmp111
    tmp115 = tmp112 & tmp114
    tmp116 = tmp115 & tmp109
    tmp117 = tl.load(in_ptr0 + ((-1) + 2*x0 + 8*x1 + 16*x2), tmp116 & xmask, eviction_policy='evict_last', other=0.0)
    tmp118 = tl.full(tmp117.shape, 0.0, tmp117.dtype)
    tmp119 = tl.where(tmp109, tmp117, tmp118)
    tmp120 = tmp119 + tmp108
    tmp121 = tmp96 & tmp40
    tmp122 = 2*x1
    tmp123 = tl.full([1], 0, tl.int64)
    tmp124 = tmp122 >= tmp123
    tmp125 = 2*x0
    tmp126 = tmp125 >= tmp123
    tmp127 = tmp124 & tmp126
    tmp128 = tmp127 & tmp121
    tmp129 = tl.load(in_ptr0 + (2*x0 + 8*x1 + 16*x2), tmp128 & xmask, eviction_policy='evict_last', other=0.0)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp121, tmp129, tmp130)
    tmp132 = tmp131 + tmp120
    tmp133 = ((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0))) + ((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5)))*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5))) + ((-1)*((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5)))) + ((-1)*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0)))*((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5))))
    tmp134 = tmp132 / tmp133
    tl.store(out_ptr0 + (x4), tmp134, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4o/c4omvc47nezolkbvcgtiimmltwssk5df2knzpjkh5ucnkf6guwkn.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_2 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 3*x1 + 9*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_constant_pad_nd_0.run(arg0_1, buf0, 144, grid=grid(144), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf0, buf1, 64, grid=grid(64), stream=stream0)
        del buf0
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
