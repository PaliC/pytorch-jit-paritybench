# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/no/cnoiof5rkg7fy5gs5juei3leueq6jm2fhffwwf3mq6uy2sxjrd5w.py
# Topologically Sorted Source Nodes: [att_signal_1, att_signal_2, att_signal_3], Original ATen: [aten.relu, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   att_signal_1 => relu
#   att_signal_2 => mul
#   att_signal_3 => sum_1
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %primals_3), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [2]), kwargs = {})
triton_poi_fused_mul_relu_sum_0 = async_compile.triton('triton_poi_fused_mul_relu_sum_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_relu_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_relu_sum_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4 + x0 + 16*x1), xmask)
    tmp9 = tl.load(in_ptr0 + (8 + x0 + 16*x1), xmask)
    tmp13 = tl.load(in_ptr0 + (12 + x0 + 16*x1), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = tmp2 * tmp3
    tmp6 = triton_helpers.maximum(tmp1, tmp5)
    tmp7 = tmp6 * tmp3
    tmp8 = tmp4 + tmp7
    tmp10 = triton_helpers.maximum(tmp1, tmp9)
    tmp11 = tmp10 * tmp3
    tmp12 = tmp8 + tmp11
    tmp14 = triton_helpers.maximum(tmp1, tmp13)
    tmp15 = tmp14 * tmp3
    tmp16 = tmp12 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ey/cey2gmm6wxtbbexmbe44arrjiylnocujhublgyggrrtcvnw6dktm.py
# Topologically Sorted Source Nodes: [att_signal_4], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   att_signal_4 => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%sum_1, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_1 = async_compile.triton('triton_poi_fused__softmax_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (4 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z2/cz2ldaeskjqttqmlqen3xepxap5e66hbiaertgoagwinyiihmzii.py
# Topologically Sorted Source Nodes: [att_signal_4], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   att_signal_4 => div, sum_2
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_2), kwargs = {})
triton_poi_fused__softmax_2 = async_compile.triton('triton_poi_fused__softmax_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (4 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [att_signal], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_2, (64, 4), (4, 1), 0), reinterpret_tensor(primals_1, (4, 4), (1, 4), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [att_signal_1, att_signal_2, att_signal_3], Original ATen: [aten.relu, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_relu_sum_0.run(buf0, primals_3, buf1, 64, grid=grid(64), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [att_signal_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf1, buf2, 64, grid=grid(64), stream=stream0)
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [att_signal_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf2, buf3, 64, grid=grid(64), stream=stream0)
        del buf2
    return (buf3, primals_3, reinterpret_tensor(primals_2, (64, 4), (4, 1), 0), buf0, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
