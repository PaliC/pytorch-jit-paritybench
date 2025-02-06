# AOT ID: ['8_forward']
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


# kernel path: inductor_cache/z2/cz2ln3hq3tlz34bqgu22tuqa6dp4bwedne32we76oebob6cl465u.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_2 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_1,), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_0 = async_compile.triton('triton_poi_fused_relu_threshold_backward_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h4/ch4iwmipufr23x3t7cam6ixhmcjvew3lj4xpsif2ogc4qmeoynpa.py
# Topologically Sorted Source Nodes: [advantage, mean], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   advantage => add
#   mean => mean
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %primals_5), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add, [-1], True), kwargs = {})
triton_poi_fused_add_mean_1 = async_compile.triton('triton_poi_fused_add_mean_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4 + tmp2
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7 + tmp2
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10 + tmp2
    tmp12 = tmp9 + tmp11
    tmp13 = 4.0
    tmp14 = tmp12 / tmp13
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tx/ctx2prb4pyjtvwhfmemumaqm22dy5ajju5ee57sexa2qovhh5ees.py
# Topologically Sorted Source Nodes: [advantage, mean, sub, add_1], Original ATen: [aten.add, aten.mean, aten.sub]
# Source node to ATen node mapping:
#   add_1 => add_1
#   advantage => add
#   mean => mean
#   sub => sub
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %primals_5), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mean), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_7, %sub), kwargs = {})
triton_poi_fused_add_mean_sub_2 = async_compile.triton('triton_poi_fused_add_mean_sub_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_sub_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp2
    tmp7 = tmp4 + tmp6
    tmp9 = tmp7 - tmp8
    tmp10 = tmp3 + tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (1, ), (1, ))
    assert_size_stride(primals_6, (4, 4), (4, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (1, 4), (4, 1))
    assert_size_stride(primals_9, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf0)
        del primals_2
        buf1 = reinterpret_tensor(buf0, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf0  # reuse
        buf9 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf1, primals_3, buf9, 256, grid=grid(256), stream=stream0)
        del primals_3
        buf2 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf2)
        buf3 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf3)
        del primals_6
        buf4 = reinterpret_tensor(buf3, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf3  # reuse
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf4, primals_7, buf8, 256, grid=grid(256), stream=stream0)
        del primals_7
        buf5 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf4, (64, 4), (4, 1), 0), reinterpret_tensor(primals_8, (4, 1), (1, 4), 0), out=buf5)
        buf6 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [advantage, mean], Original ATen: [aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_1.run(buf2, primals_5, buf6, 64, grid=grid(64), stream=stream0)
        buf7 = reinterpret_tensor(buf2, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [advantage, mean, sub, add_1], Original ATen: [aten.add, aten.mean, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_sub_2.run(buf7, buf5, primals_9, primals_5, buf6, 256, grid=grid(256), stream=stream0)
        del buf5
        del buf6
        del primals_5
        del primals_9
    return (buf7, reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), reinterpret_tensor(buf1, (64, 4), (4, 1), 0), reinterpret_tensor(buf4, (64, 4), (4, 1), 0), primals_8, buf8, primals_4, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
