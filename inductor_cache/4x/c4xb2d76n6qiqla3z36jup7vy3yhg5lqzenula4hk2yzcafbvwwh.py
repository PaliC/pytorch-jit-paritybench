# AOT ID: ['4_forward']
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


# kernel path: inductor_cache/6l/c6lkl32qktqe2so7ghbrha3a52qcdmhew7rqzybvog57z22gqwmc.py
# Topologically Sorted Source Nodes: [out_2, pow_2, out_3], Original ATen: [aten.maximum, aten.pow, aten.sub]
# Source node to ATen node mapping:
#   out_2 => maximum_1
#   out_3 => sub_1
#   pow_2 => pow_2
# Graph fragment:
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%primals_5, %primals_6), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum_1, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_2, %primals_7), kwargs = {})
triton_poi_fused_maximum_pow_sub_0 = async_compile.triton('triton_poi_fused_maximum_pow_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_maximum_pow_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_maximum_pow_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp3 = triton_helpers.maximum(tmp0, tmp2)
    tmp4 = tmp3 * tmp3
    tmp7 = tmp4 - tmp6
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mc/cmcqvbrkofvsyf5l72wcumgrmjvltk7fv56uwr6inmqbyyo4acc7.py
# Topologically Sorted Source Nodes: [abs_1], Original ATen: [aten.abs]
# Source node to ATen node mapping:
#   abs_1 => abs_1
# Graph fragment:
#   %abs_1 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%primals_1,), kwargs = {})
triton_poi_fused_abs_1 = async_compile.triton('triton_poi_fused_abs_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl_math.abs(tmp0)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hn/chn2wwbnzasrokpurdx4dvcetdug3em74fo63wl3dkpfumgwr7fb.py
# Topologically Sorted Source Nodes: [out, pow_1, out_1, norm], Original ATen: [aten.maximum, aten.pow, aten.sub, aten.convolution]
# Source node to ATen node mapping:
#   norm => convolution
#   out => maximum
#   out_1 => sub
#   pow_1 => pow_1
# Graph fragment:
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%primals_2, %primals_3), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_1, %primals_4), kwargs = {})
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%abs_1, %view, %sub, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_maximum_pow_sub_2 = async_compile.triton('triton_poi_fused_convolution_maximum_pow_sub_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_maximum_pow_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_maximum_pow_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp3 = triton_helpers.maximum(tmp0, tmp2)
    tmp4 = tmp3 * tmp3
    tmp7 = tmp4 - tmp6
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mz/cmzh4c5wbe27bfktsn7qcwohuprd7xry7xhc5qgoofr2vitovqus.py
# Topologically Sorted Source Nodes: [out, pow_1, out_1, norm, norm_1, out_4], Original ATen: [aten.maximum, aten.pow, aten.sub, aten.convolution, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   norm => convolution
#   norm_1 => mul, reciprocal
#   out => maximum
#   out_1 => sub
#   out_4 => mul_1
#   pow_1 => pow_1
# Graph fragment:
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%primals_2, %primals_3), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_1, %primals_4), kwargs = {})
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%abs_1, %view, %sub, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convolution,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %mul), kwargs = {})
triton_poi_fused_convolution_maximum_mul_pow_reciprocal_sub_3 = async_compile.triton('triton_poi_fused_convolution_maximum_mul_pow_reciprocal_sub_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_maximum_mul_pow_reciprocal_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_maximum_mul_pow_reciprocal_sub_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp4 / tmp2
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 * tmp7
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (1, ), (1, ))
    assert_size_stride(primals_4, (1, ), (1, ))
    assert_size_stride(primals_5, (4, 4), (4, 1))
    assert_size_stride(primals_6, (1, ), (1, ))
    assert_size_stride(primals_7, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2, pow_2, out_3], Original ATen: [aten.maximum, aten.pow, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_maximum_pow_sub_0.run(primals_5, primals_6, primals_7, buf0, 16, grid=grid(16), stream=stream0)
        del primals_7
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [abs_1], Original ATen: [aten.abs]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_1.run(primals_1, buf1, 256, grid=grid(256), stream=stream0)
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out, pow_1, out_1, norm], Original ATen: [aten.maximum, aten.pow, aten.sub, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_maximum_pow_sub_2.run(primals_2, primals_3, primals_4, buf2, 4, grid=grid(4), stream=stream0)
        del primals_4
        # Topologically Sorted Source Nodes: [out, pow_1, out_1, norm], Original ATen: [aten.maximum, aten.pow, aten.sub, aten.convolution]
        buf3 = extern_kernels.convolution(buf1, reinterpret_tensor(buf0, (4, 4, 1, 1), (4, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 4, 4), (64, 16, 4, 1))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out, pow_1, out_1, norm, norm_1, out_4], Original ATen: [aten.maximum, aten.pow, aten.sub, aten.convolution, aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_maximum_mul_pow_reciprocal_sub_3.run(buf4, buf2, primals_1, buf5, 256, grid=grid(256), stream=stream0)
        del buf2
    return (buf5, primals_1, primals_2, primals_3, primals_5, primals_6, reinterpret_tensor(buf0, (4, 4, 1, 1), (4, 1, 1, 1), 0), buf1, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
