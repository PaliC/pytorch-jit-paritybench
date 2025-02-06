# AOT ID: ['4_inference']
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


# kernel path: inductor_cache/lf/clf4wjkxm5yywupqgmh2t2pbf2yfiocbvrpdmaekiikjnjkdw5cz.py
# Topologically Sorted Source Nodes: [getitem, clamp, setitem], Original ATen: [aten.index, aten.clamp, aten.index_put]
# Source node to ATen node mapping:
#   clamp => clamp_min
#   getitem => index
#   setitem => index_put
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%lift_fresh_copy]), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%index, 0), kwargs = {})
#   %index_put : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%arg0_1, [%lift_fresh_copy_1], %clamp_min), kwargs = {})
triton_poi_fused_clamp_index_index_put_0 = async_compile.triton('triton_poi_fused_clamp_index_index_put_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_index_index_put_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_index_index_put_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6e/c6eakbqz4ptk6jltbw4bzz4eefpxtyla6wxcutn5fqh45zighhmn.py
# Topologically Sorted Source Nodes: [getitem, clamp, setitem], Original ATen: [aten.index, aten.clamp, aten.index_put]
# Source node to ATen node mapping:
#   clamp => clamp_min
#   getitem => index
#   setitem => index_put
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%lift_fresh_copy]), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%index, 0), kwargs = {})
#   %index_put : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%arg0_1, [%lift_fresh_copy_1], %clamp_min), kwargs = {})
triton_poi_fused_clamp_index_index_put_1 = async_compile.triton('triton_poi_fused_clamp_index_index_put_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_index_index_put_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_index_index_put_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.load(in_ptr0 + (x0 + 64*tmp5), xmask)
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tl.store(out_ptr0 + (x0 + 64*tmp5), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4j/c4jurxa32by5wtwzp43iyy3tvi5a3bbfgii7v3avudjh4ojccdep.py
# Topologically Sorted Source Nodes: [getitem_1, sub, setitem_1], Original ATen: [aten.index, aten.rsub, aten.index_put]
# Source node to ATen node mapping:
#   getitem_1 => index_1
#   setitem_1 => index_put_1
#   sub => sub
# Graph fragment:
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%index_put, [%lift_fresh_copy_2]), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (0, %index_1), kwargs = {})
#   %index_put_1 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put, [%lift_fresh_copy_3], %sub), kwargs = {})
triton_poi_fused_index_index_put_rsub_2 = async_compile.triton('triton_poi_fused_index_index_put_rsub_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_put_rsub_2', 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_index_put_rsub_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 3, tl.int64)
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.load(in_ptr0 + (x0 + 64*tmp4), xmask)
    tmp6 = 0.0
    tmp7 = tmp6 - tmp5
    tl.store(out_ptr0 + (x0 + 64*tmp4), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yc/cyc5xslapwpvcvxonq6xjtgqkom6wkv4kvp2jnfexwgtks3a6mzq.py
# Topologically Sorted Source Nodes: [getitem_2, clamp_1, neg, setitem_2], Original ATen: [aten.index, aten.clamp, aten.neg, aten.index_put]
# Source node to ATen node mapping:
#   clamp_1 => clamp_min_1
#   getitem_2 => index_2
#   neg => neg
#   setitem_2 => index_put_2
# Graph fragment:
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%index_put_1, [%lift_fresh_copy_4]), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%index_2, 0), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%clamp_min_1,), kwargs = {})
#   %index_put_2 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_1, [%lift_fresh_copy_5], %neg), kwargs = {})
triton_poi_fused_clamp_index_index_put_neg_3 = async_compile.triton('triton_poi_fused_clamp_index_index_put_neg_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_index_index_put_neg_3', 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_index_index_put_neg_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 3, tl.int64)
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.load(in_ptr0 + (x0 + 64*tmp4), xmask)
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = -tmp7
    tl.store(out_ptr0 + (x0 + 64*tmp4), tmp8, xmask)
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
        # Topologically Sorted Source Nodes: [getitem, clamp, setitem], Original ATen: [aten.index, aten.clamp, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_index_index_put_0.run(arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [getitem, clamp, setitem], Original ATen: [aten.index, aten.clamp, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_index_index_put_1.run(arg0_1, buf0, 128, grid=grid(128), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [getitem_1, sub, setitem_1], Original ATen: [aten.index, aten.rsub, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_index_put_rsub_2.run(buf0, buf0, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [getitem_2, clamp_1, neg, setitem_2], Original ATen: [aten.index, aten.clamp, aten.neg, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_index_index_put_neg_3.run(buf0, buf0, 128, grid=grid(128), stream=stream0)
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
