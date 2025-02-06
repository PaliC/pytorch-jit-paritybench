# AOT ID: ['65_inference']
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


# kernel path: inductor_cache/bs/cbsicsqji23fdxzjlziry3a6hhdh64kajei6ficiqtypdite3foo.py
# Topologically Sorted Source Nodes: [embedding, embedding_1, add, embedding_2, add_1, embedding_3, add_2, add_3], Original ATen: [aten.embedding, aten.add]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   embedding => embedding
#   embedding_1 => embedding_1
#   embedding_2 => embedding_2
#   embedding_3 => embedding_3
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %select), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %select_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %select_2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %embedding_3 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %select_3), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %embedding_3), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 0.0), kwargs = {})
triton_poi_fused_add_embedding_0 = async_compile.triton('triton_poi_fused_add_embedding_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_embedding_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x0 = (xindex % 4)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (12 + x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (8 + x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (4 + x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([XBLOCK], 24, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 24)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 24")
    tmp7 = tl.load(in_ptr1 + (x0 + 4*tmp5), xmask)
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tl.full([XBLOCK], 7, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert(((0 <= tmp13) & (tmp13 < 7)) | ~(xmask), "index out of bounds: 0 <= tmp13 < 7")
    tmp15 = tl.load(in_ptr2 + (x0 + 4*tmp13), xmask)
    tmp16 = tmp7 + tmp15
    tmp18 = tmp17.to(tl.int64)
    tmp19 = tl.full([XBLOCK], 32, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert(((0 <= tmp22) & (tmp22 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp22 < 32")
    tmp24 = tl.load(in_ptr3 + (x0 + 4*tmp22), xmask)
    tmp25 = tmp16 + tmp24
    tmp27 = tmp26.to(tl.int64)
    tmp28 = tl.full([XBLOCK], 13, tl.int32)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp27 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp27)
    tl.device_assert(((0 <= tmp31) & (tmp31 < 13)) | ~(xmask), "index out of bounds: 0 <= tmp31 < 13")
    tmp33 = tl.load(in_ptr4 + (x0 + 4*tmp31), xmask)
    tmp34 = tmp25 + tmp33
    tmp35 = 0.0
    tmp36 = tmp34 + tmp35
    tl.store(out_ptr0 + (x4), tmp36, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (24, 4), (4, 1))
    assert_size_stride(arg2_1, (7, 4), (4, 1))
    assert_size_stride(arg3_1, (32, 4), (4, 1))
    assert_size_stride(arg4_1, (13, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, embedding_1, add, embedding_2, add_1, embedding_3, add_2, add_3], Original ATen: [aten.embedding, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_embedding_0.run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((24, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((7, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((13, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
