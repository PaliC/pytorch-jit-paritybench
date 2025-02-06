# AOT ID: ['35_forward']
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


# kernel path: inductor_cache/j5/cj5chulcbvuhgvrwzbxwpz7ioekz5mw7x3bgdias7evui5ajdrri.py
# Topologically Sorted Source Nodes: [padded], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   padded => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad1d_0 = async_compile.triton('triton_poi_fused_reflection_pad1d_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 10)
    x1 = xindex // 10
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + ((-1)*tl_math.abs((-3) + tl_math.abs((-3) + x0))) + 4*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hz/chz3jli66pfee32a42dvt6gcbdxlv7r3kxlupfrbz4c5apkc5xmj.py
# Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm => pow_1, pow_2, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_3, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
triton_poi_fused__weight_norm_interface_1 = async_compile.triton('triton_poi_fused__weight_norm_interface_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tl.store(out_ptr0 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gr/cgrylhb7mbyxgh3n7mlc6oib6xw6j2vinwsgtbad5ieix6a6qsve.py
# Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm => div, mul
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_2, %pow_2), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, %div), kwargs = {})
triton_poi_fused__weight_norm_interface_2 = async_compile.triton('triton_poi_fused__weight_norm_interface_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 7
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ha/cha33xqe6vik7gwjip3lxfxchtztrxtlj6pnxpysdf2akk7zypw6.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_1 => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index, %mul, %primals_4, [1], [0], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (4, 1, 4), (4, 4, 1))
    assert_size_stride(primals_2, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_3, (32, 1, 7), (7, 7, 1))
    assert_size_stride(primals_4, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 10), (10, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [padded], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_0.run(primals_1, buf0, 40, grid=grid(40), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((32, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_1.run(primals_3, buf1, 32, grid=grid(32), stream=stream0)
        buf2 = empty_strided_cuda((32, 1, 7), (7, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_2.run(primals_3, primals_2, buf1, buf2, 224, grid=grid(224), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf0, buf2, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf3, (4, 32, 4), (128, 4, 1))
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf4, primals_4, 512, grid=grid(512), stream=stream0)
        del primals_4
    return (buf4, buf2, primals_2, primals_3, buf0, buf1, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 1, 4), (4, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 1, 7), (7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
