# AOT ID: ['5_inference']
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


# kernel path: inductor_cache/rq/crqaslcpxb4eqwbbvnv3rljk6j73fhug7r433zgeolwymvo3v5wj.py
# Topologically Sorted Source Nodes: [max_pool2d_2, features], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
# Source node to ATen node mapping:
#   features => cat
#   max_pool2d_2 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%arg0_1, [5, 5], [1, 1], [2, 2], [1, 1], False), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %getitem_2, %getitem_4, %arg0_1], 1), kwargs = {})
triton_poi_fused_cat_max_pool2d_with_indices_0 = async_compile.triton('triton_poi_fused_cat_max_pool2d_with_indices_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x7 = xindex
    x3 = xindex // 64
    x4 = (xindex % 64)
    tmp116 = tl.load(in_ptr0 + (x7), xmask)
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-2) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-10) + x7), tmp10 & xmask, other=float("-inf"))
    tmp12 = (-1) + x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-9) + x7), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8) + x7), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 1 + x0
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tl.load(in_ptr0 + ((-7) + x7), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = 2 + x0
    tmp34 = tmp33 >= tmp1
    tmp35 = tmp33 < tmp3
    tmp36 = tmp34 & tmp35
    tmp37 = tmp5 & tmp36
    tmp38 = tl.load(in_ptr0 + ((-6) + x7), tmp37 & xmask, other=float("-inf"))
    tmp39 = triton_helpers.maximum(tmp38, tmp32)
    tmp40 = (-1) + x1
    tmp41 = tmp40 >= tmp1
    tmp42 = tmp40 < tmp3
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp9
    tmp45 = tl.load(in_ptr0 + ((-6) + x7), tmp44 & xmask, other=float("-inf"))
    tmp46 = triton_helpers.maximum(tmp45, tmp39)
    tmp47 = tmp43 & tmp15
    tmp48 = tl.load(in_ptr0 + ((-5) + x7), tmp47 & xmask, other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp46)
    tmp50 = tmp43 & tmp22
    tmp51 = tl.load(in_ptr0 + ((-4) + x7), tmp50 & xmask, other=float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp49)
    tmp53 = tmp43 & tmp29
    tmp54 = tl.load(in_ptr0 + ((-3) + x7), tmp53 & xmask, other=float("-inf"))
    tmp55 = triton_helpers.maximum(tmp54, tmp52)
    tmp56 = tmp43 & tmp36
    tmp57 = tl.load(in_ptr0 + ((-2) + x7), tmp56 & xmask, other=float("-inf"))
    tmp58 = triton_helpers.maximum(tmp57, tmp55)
    tmp59 = x1
    tmp60 = tmp59 >= tmp1
    tmp61 = tmp59 < tmp3
    tmp62 = tmp60 & tmp61
    tmp63 = tmp62 & tmp9
    tmp64 = tl.load(in_ptr0 + ((-2) + x7), tmp63 & xmask, other=float("-inf"))
    tmp65 = triton_helpers.maximum(tmp64, tmp58)
    tmp66 = tmp62 & tmp15
    tmp67 = tl.load(in_ptr0 + ((-1) + x7), tmp66 & xmask, other=float("-inf"))
    tmp68 = triton_helpers.maximum(tmp67, tmp65)
    tmp69 = tmp62 & tmp22
    tmp70 = tl.load(in_ptr0 + (x7), tmp69 & xmask, other=float("-inf"))
    tmp71 = triton_helpers.maximum(tmp70, tmp68)
    tmp72 = tmp62 & tmp29
    tmp73 = tl.load(in_ptr0 + (1 + x7), tmp72 & xmask, other=float("-inf"))
    tmp74 = triton_helpers.maximum(tmp73, tmp71)
    tmp75 = tmp62 & tmp36
    tmp76 = tl.load(in_ptr0 + (2 + x7), tmp75 & xmask, other=float("-inf"))
    tmp77 = triton_helpers.maximum(tmp76, tmp74)
    tmp78 = 1 + x1
    tmp79 = tmp78 >= tmp1
    tmp80 = tmp78 < tmp3
    tmp81 = tmp79 & tmp80
    tmp82 = tmp81 & tmp9
    tmp83 = tl.load(in_ptr0 + (2 + x7), tmp82 & xmask, other=float("-inf"))
    tmp84 = triton_helpers.maximum(tmp83, tmp77)
    tmp85 = tmp81 & tmp15
    tmp86 = tl.load(in_ptr0 + (3 + x7), tmp85 & xmask, other=float("-inf"))
    tmp87 = triton_helpers.maximum(tmp86, tmp84)
    tmp88 = tmp81 & tmp22
    tmp89 = tl.load(in_ptr0 + (4 + x7), tmp88 & xmask, other=float("-inf"))
    tmp90 = triton_helpers.maximum(tmp89, tmp87)
    tmp91 = tmp81 & tmp29
    tmp92 = tl.load(in_ptr0 + (5 + x7), tmp91 & xmask, other=float("-inf"))
    tmp93 = triton_helpers.maximum(tmp92, tmp90)
    tmp94 = tmp81 & tmp36
    tmp95 = tl.load(in_ptr0 + (6 + x7), tmp94 & xmask, other=float("-inf"))
    tmp96 = triton_helpers.maximum(tmp95, tmp93)
    tmp97 = 2 + x1
    tmp98 = tmp97 >= tmp1
    tmp99 = tmp97 < tmp3
    tmp100 = tmp98 & tmp99
    tmp101 = tmp100 & tmp9
    tmp102 = tl.load(in_ptr0 + (6 + x7), tmp101 & xmask, other=float("-inf"))
    tmp103 = triton_helpers.maximum(tmp102, tmp96)
    tmp104 = tmp100 & tmp15
    tmp105 = tl.load(in_ptr0 + (7 + x7), tmp104 & xmask, other=float("-inf"))
    tmp106 = triton_helpers.maximum(tmp105, tmp103)
    tmp107 = tmp100 & tmp22
    tmp108 = tl.load(in_ptr0 + (8 + x7), tmp107 & xmask, other=float("-inf"))
    tmp109 = triton_helpers.maximum(tmp108, tmp106)
    tmp110 = tmp100 & tmp29
    tmp111 = tl.load(in_ptr0 + (9 + x7), tmp110 & xmask, other=float("-inf"))
    tmp112 = triton_helpers.maximum(tmp111, tmp109)
    tmp113 = tmp100 & tmp36
    tmp114 = tl.load(in_ptr0 + (10 + x7), tmp113 & xmask, other=float("-inf"))
    tmp115 = triton_helpers.maximum(tmp114, tmp112)
    tl.store(out_ptr0 + (x4 + 256*x3), tmp115, xmask)
    tl.store(out_ptr1 + (x4 + 256*x3), tmp116, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vl/cvll6xb6s4i2i42jlxr7vqrawy6a2izdqx6ck75h6usp6u5pauzc.py
# Topologically Sorted Source Nodes: [features], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   features => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %getitem_2, %getitem_4, %arg0_1], 1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 256*x1), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [max_pool2d], Original ATen: [aten.max_pool2d_with_indices]
        buf0 = torch.ops.aten.max_pool2d_with_indices.default(arg0_1, [13, 13], [1, 1], [6, 6])
        buf1 = buf0[0]
        del buf0
        # Topologically Sorted Source Nodes: [max_pool2d_1], Original ATen: [aten.max_pool2d_with_indices]
        buf3 = torch.ops.aten.max_pool2d_with_indices.default(arg0_1, [9, 9], [1, 1], [4, 4])
        buf4 = buf3[0]
        del buf3
        buf10 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        buf6 = reinterpret_tensor(buf10, (4, 4, 4, 4), (256, 16, 4, 1), 128)  # alias
        buf9 = reinterpret_tensor(buf10, (4, 4, 4, 4), (256, 16, 4, 1), 192)  # alias
        # Topologically Sorted Source Nodes: [max_pool2d_2, features], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_max_pool2d_with_indices_0.run(arg0_1, buf6, buf9, 256, grid=grid(256), stream=stream0)
        del arg0_1
        buf7 = reinterpret_tensor(buf10, (4, 4, 4, 4), (256, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [features], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf1, buf7, 256, grid=grid(256), stream=stream0)
        del buf1
        buf8 = reinterpret_tensor(buf10, (4, 4, 4, 4), (256, 16, 4, 1), 64)  # alias
        # Topologically Sorted Source Nodes: [features], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf4, buf8, 256, grid=grid(256), stream=stream0)
        del buf4
    return (buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
