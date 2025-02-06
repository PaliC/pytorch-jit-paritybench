# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/y3/cy3m3pau3hkp4ifn6enl725nxnwsif2hmh7sd4szoxoux6kvciza.py
# Topologically Sorted Source Nodes: [pow_1, avg_pool1d], Original ATen: [aten.pow, aten.unsqueeze]
# Source node to ATen node mapping:
#   avg_pool1d => unsqueeze
#   pow_1 => pow_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%permute, 2), kwargs = {})
#   %unsqueeze : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%pow_1, -2), kwargs = {})
triton_poi_fused_pow_unsqueeze_0 = async_compile.triton('triton_poi_fused_pow_unsqueeze_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_pow_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_pow_unsqueeze_0(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 8
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 65536*x1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tl.store(out_ptr1 + (y0 + 65536*x1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/2p/c2pnktqqchrbio2qz45c5usxm5fhpvg64sehy75lkhcjwelnieka.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   x_3 => mul
#   x_4 => permute_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 2), kwargs = {})
#   %permute_1 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
triton_poi_fused_mul_transpose_1 = async_compile.triton('triton_poi_fused_mul_transpose_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_transpose_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_transpose_1(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 131072*y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (65536 + x1 + 131072*y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x1 + 65536*y0), tmp6, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/dr/cdrpqpn7hxhn2u6lvf2wrzp7q6kexgvbnzeda5icr5z3hu346njs.py
# Topologically Sorted Source Nodes: [x_6, x_7, x_8], Original ATen: [aten.abs, aten.add, aten.log]
# Source node to ATen node mapping:
#   x_6 => abs_1
#   x_7 => add
#   x_8 => log
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convolution_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, 1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
triton_poi_fused_abs_add_log_2 = async_compile.triton('triton_poi_fused_abs_add_log_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_log_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_log_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = tl_math.log(tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(primals_2, (8, 1, 401), (401, 401, 1))
    assert_size_stride(primals_3, (4, 1, 401), (401, 401, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_1, (1, 1, 65536), (65536, 65536, 1), 0), primals_2, stride=(1,), padding=(200,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf0, (1, 8, 65536), (524288, 65536, 1))
        buf2 = empty_strided_cuda((1, 65536, 1, 8), (524288, 1, 524288, 65536), torch.float32)
        # Topologically Sorted Source Nodes: [pow_1, avg_pool1d], Original ATen: [aten.pow, aten.unsqueeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_unsqueeze_0.run(buf0, buf2, 65536, 8, grid=grid(65536, 8), stream=stream0)
        buf4 = empty_strided_cuda((1, 4, 65536), (262144, 65536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.mul, aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_transpose_1.run(buf2, buf4, 4, 65536, grid=grid(4, 65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_3, stride=(160,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf5, (1, 4, 408), (1632, 408, 1))
        buf6 = empty_strided_cuda((1, 4, 408), (1632, 408, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, x_7, x_8], Original ATen: [aten.abs, aten.add, aten.log]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_log_2.run(buf5, buf6, 1632, grid=grid(1632), stream=stream0)
    return (buf6, primals_2, primals_3, reinterpret_tensor(primals_1, (1, 1, 65536), (65536, 65536, 1), 0), buf0, buf2, buf4, buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((8, 1, 401), (401, 401, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 1, 401), (401, 401, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
