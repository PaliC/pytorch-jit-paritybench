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


# kernel path: inductor_cache/hq/chqlc5qqmtpj5bdqluehpwl6lvzjdhdnkuq55vsmjtb5prnwk2ja.py
# Topologically Sorted Source Nodes: [pos], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   pos => iota
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
triton_poi_fused_arange_0 = async_compile.triton('triton_poi_fused_arange_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r4/cr4djndwypjorxar65ftydcjpk3p6cb23qddqmv6uwnk3t5snrgh.py
# Topologically Sorted Source Nodes: [embedding, e, u], Original ATen: [aten.embedding, aten.add, aten.mean]
# Source node to ATen node mapping:
#   e => add
#   embedding => embedding
#   u => mean
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %expand), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %embedding), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add, [-1], True), kwargs = {})
triton_poi_fused_add_embedding_mean_1 = async_compile.triton('triton_poi_fused_add_embedding_mean_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mean_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_embedding_mean_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([XBLOCK], 4, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 4")
    tmp7 = tl.load(in_ptr2 + (4*tmp5), xmask, eviction_policy='evict_last')
    tmp8 = tmp0 + tmp7
    tmp10 = tl.load(in_ptr2 + (1 + 4*tmp5), xmask, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp14 = tl.load(in_ptr2 + (2 + 4*tmp5), xmask, eviction_policy='evict_last')
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 + tmp15
    tmp18 = tl.load(in_ptr2 + (3 + 4*tmp5), xmask, eviction_policy='evict_last')
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = 4.0
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/df/cdfjb5wivun4f2ogkszh43hstj35j5jketx7amwkbn3mf4tkfln4.py
# Topologically Sorted Source Nodes: [embedding, e, sub], Original ATen: [aten.embedding, aten.add, aten.sub]
# Source node to ATen node mapping:
#   e => add
#   embedding => embedding
#   sub => sub
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %expand), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %embedding), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mean), kwargs = {})
triton_poi_fused_add_embedding_sub_2 = async_compile.triton('triton_poi_fused_add_embedding_sub_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_embedding_sub_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x4 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([XBLOCK], 4, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 4")
    tmp7 = tl.load(in_ptr2 + (x0 + 4*tmp5), xmask)
    tmp8 = tmp0 + tmp7
    tmp10 = tmp8 - tmp9
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lh/clhmd63srmxbfhnnz3iwflertjnyyeqyzsexbw4uzzouat3ykn5v.py
# Topologically Sorted Source Nodes: [pow_1, s, add_1, sqrt, x, mul, add_2], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_2 => add_2
#   mul => mul
#   pow_1 => pow_1
#   s => mean_1
#   sqrt => sqrt
#   x => div
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-12), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %sqrt), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, %div), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %primals_4), kwargs = {})
triton_poi_fused_add_div_mean_mul_pow_sqrt_3 = async_compile.triton('triton_poi_fused_add_div_mean_mul_pow_sqrt_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_mul_pow_sqrt_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_mul_pow_sqrt_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp2 * tmp2
    tmp5 = tmp4 * tmp4
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = 4.0
    tmp14 = tmp12 / tmp13
    tmp15 = 1e-12
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tmp1 / tmp17
    tmp19 = tmp0 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pos], Original ATen: [aten.arange]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_0.run(buf0, 4, grid=grid(4), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, e, u], Original ATen: [aten.embedding, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_embedding_mean_1.run(primals_1, buf0, primals_2, buf1, 16, grid=grid(16), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, e, sub], Original ATen: [aten.embedding, aten.add, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_embedding_sub_2.run(primals_1, buf0, primals_2, buf1, buf2, 64, grid=grid(64), stream=stream0)
        del buf1
        del primals_1
        del primals_2
        buf3 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_1, s, add_1, sqrt, x, mul, add_2], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_mul_pow_sqrt_3.run(primals_3, buf2, primals_4, buf3, 64, grid=grid(64), stream=stream0)
        del primals_4
    return (buf3, primals_3, reinterpret_tensor(buf0, (1, 4), (4, 1), 0), buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
