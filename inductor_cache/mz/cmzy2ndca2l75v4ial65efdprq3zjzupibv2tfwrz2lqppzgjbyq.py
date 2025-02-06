# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/6y/c6ybadripvlz7i5ihjmmn4xmkahw3dzo664iqrd5aic2rmmf5ha6.py
# Topologically Sorted Source Nodes: [sub, norm, first_element, abs_1, sub_2, abs_2, second_element, truediv, truediv_1, sub_3], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.add, aten.abs, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   abs_1 => abs_2
#   abs_2 => abs_4
#   first_element => add
#   norm => abs_1, pow_2, sum_1
#   second_element => add_1
#   sub => sub
#   sub_2 => sub_2
#   sub_3 => sub_3
#   truediv => mul, reciprocal
#   truediv_1 => mul_1, reciprocal_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %primals_2), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [1]), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 1.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%primals_3,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_2, %pow_2), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_2,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_4, 1), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %mul_1), kwargs = {})
triton_poi_fused_abs_add_linalg_vector_norm_mul_reciprocal_sub_0 = async_compile.triton('triton_poi_fused_abs_add_linalg_vector_norm_mul_reciprocal_sub_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_linalg_vector_norm_mul_reciprocal_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_linalg_vector_norm_mul_reciprocal_sub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16
    x4 = (xindex % 16)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x4 + 64*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (16 + x4 + 64*x2), xmask)
    tmp8 = tl.load(in_ptr0 + (32 + x4 + 64*x2), xmask)
    tmp12 = tl.load(in_ptr0 + (48 + x4 + 64*x2), xmask)
    tmp21 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp5 = tmp4 - tmp1
    tmp6 = tl_math.abs(tmp5)
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8 - tmp1
    tmp10 = tl_math.abs(tmp9)
    tmp11 = tmp7 + tmp10
    tmp13 = tmp12 - tmp1
    tmp14 = tl_math.abs(tmp13)
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp18 / tmp17
    tmp20 = tmp19 * tmp16
    tmp22 = tl_math.abs(tmp21)
    tmp23 = tmp22 - tmp15
    tmp24 = tl_math.abs(tmp23)
    tmp25 = tmp24 + tmp16
    tmp26 = tmp18 / tmp25
    tmp27 = tmp26 * tmp16
    tmp28 = tmp20 - tmp27
    tl.store(in_out_ptr0 + (x5), tmp28, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (1, 4), (4, 1))
    assert_size_stride(primals_3, (1, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        buf1 = reinterpret_tensor(buf0, (4, 1, 4, 4), (16, 16, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [sub, norm, first_element, abs_1, sub_2, abs_2, second_element, truediv, truediv_1, sub_3], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.add, aten.abs, aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_linalg_vector_norm_mul_reciprocal_sub_0.run(buf1, primals_1, primals_2, primals_3, 64, grid=grid(64), stream=stream0)
    return (buf1, primals_1, primals_2, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
