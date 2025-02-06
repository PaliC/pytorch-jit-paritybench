# AOT ID: ['18_forward']
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


# kernel path: inductor_cache/7k/c7k6es557lozskfvfgjolsyzc6qpqpr4n2atoszc4uhraoxk6jl7.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.int64), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.int64)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/or/cor6fixcbjnb3433zf5wc63ynlhrbewyspnouiregpes2kuq6ct7.py
# Topologically Sorted Source Nodes: [hour_x, weekday_x, day_x, month_x, add, add_1, add_2, x_tmp_embed], Original ATen: [aten.embedding, aten.add]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   day_x => embedding_3
#   hour_x => embedding_1
#   month_x => embedding_4
#   weekday_x => embedding_2
#   x_tmp_embed => add_3
# Graph fragment:
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_4, %select), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_5, %select_1), kwargs = {})
#   %embedding_3 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_6, %select_2), kwargs = {})
#   %embedding_4 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_7, %select_3), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding_1, %embedding_2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_3), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %embedding_4), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 0.0), kwargs = {})
triton_poi_fused_add_embedding_1 = async_compile.triton('triton_poi_fused_add_embedding_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_embedding_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x0 = (xindex % 4)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (12 + x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8 + x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (4 + x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 24, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 24)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 24")
    tmp6 = tl.load(in_ptr1 + (x0 + 4*tmp4), xmask)
    tmp8 = tl.full([XBLOCK], 7, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 7)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 7")
    tmp13 = tl.load(in_ptr2 + (x0 + 4*tmp11), xmask)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([XBLOCK], 32, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 32")
    tmp21 = tl.load(in_ptr3 + (x0 + 4*tmp19), xmask)
    tmp22 = tmp14 + tmp21
    tmp24 = tl.full([XBLOCK], 13, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 13)) | ~(xmask), "index out of bounds: 0 <= tmp27 < 13")
    tmp29 = tl.load(in_ptr4 + (x0 + 4*tmp27), xmask)
    tmp30 = tmp22 + tmp29
    tmp31 = 0.0
    tmp32 = tmp30 + tmp31
    tl.store(out_ptr0 + (x4), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/re/crecp3h4yqejd6xc6dfrxghqhke46ql2mochbh7hovfgti2tna3k.py
# Topologically Sorted Source Nodes: [spa_emb, sigmoid, mul, sigmoid_1, mul_1, x_st], Original ATen: [aten.embedding, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul => mul
#   mul_1 => mul_1
#   sigmoid => sigmoid
#   sigmoid_1 => sigmoid_1
#   spa_emb => embedding
#   x_st => mul_2
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %convert_element_type), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%squeeze,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, %sigmoid), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %mul_1), kwargs = {})
triton_poi_fused_embedding_mul_sigmoid_2 = async_compile.triton('triton_poi_fused_embedding_mul_sigmoid_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_mul_sigmoid_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_mul_sigmoid_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x4 = xindex
    x2 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (x0 + 4*tmp4), xmask)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 * tmp11
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp12, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (24, 4), (4, 1))
    assert_size_stride(primals_5, (7, 4), (4, 1))
    assert_size_stride(primals_6, (32, 4), (4, 1))
    assert_size_stride(primals_7, (13, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_1, buf0, 256, grid=grid(256), stream=stream0)
        del primals_1
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_3, buf2, 256, grid=grid(256), stream=stream0)
        del primals_3
        buf3 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hour_x, weekday_x, day_x, month_x, add, add_1, add_2, x_tmp_embed], Original ATen: [aten.embedding, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_embedding_1.run(buf2, primals_4, primals_5, primals_6, primals_7, buf3, 256, grid=grid(256), stream=stream0)
        del primals_4
        del primals_5
        del primals_6
        del primals_7
        buf1 = empty_strided_cuda((4, 4, 4, 4, 4), (256, 64, 16, 4, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 4, 4, 4, 4), (256, 64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [spa_emb, sigmoid, mul, sigmoid_1, mul_1, x_st], Original ATen: [aten.embedding, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_mul_sigmoid_2.run(buf0, primals_2, buf3, buf1, buf4, 1024, grid=grid(1024), stream=stream0)
        del primals_2
    return (buf4, buf0, buf1, reinterpret_tensor(buf2, (4, 4, 4), (64, 16, 1), 12), reinterpret_tensor(buf2, (4, 4, 4), (64, 16, 1), 8), reinterpret_tensor(buf2, (4, 4, 4), (64, 16, 1), 4), reinterpret_tensor(buf2, (4, 4, 4), (64, 16, 1), 0), buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((7, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((13, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
