# AOT ID: ['7_forward']
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


# kernel path: inductor_cache/t7/ct77jvlu7n62uvqb5mtr35ghtc6h634klswijhn4nrgw25ybzmou.py
# Topologically Sorted Source Nodes: [exp, mul, z], Original ATen: [aten.exp, aten.mul, aten.add]
# Source node to ATen node mapping:
#   exp => exp
#   mul => mul
#   z => add
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%primals_2,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %randn), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %mul), kwargs = {})
triton_poi_fused_add_exp_mul_0 = async_compile.triton('triton_poi_fused_add_exp_mul_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = tl_math.exp(tmp1)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hf/chfbx37mn3kce5aybannls6n4zq2rcmehogrz55gy3jfa7peanwx.py
# Topologically Sorted Source Nodes: [wrapped_mul, wrapped_log, wrapped_mul_1, pow_1, mul_1, add_1, sum_1, log_p], Original ATen: [aten.lift_fresh, aten.mul, aten.log, aten.pow, aten.add, aten.sum, aten.sub]
# Source node to ATen node mapping:
#   add_1 => add_1
#   log_p => sub
#   mul_1 => mul_3
#   pow_1 => pow_1
#   sum_1 => sum_1
#   wrapped_log => full_default_1
#   wrapped_mul => full_default, mul_1
#   wrapped_mul_1 => mul_2
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -0.5), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %primals_3), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.8378770664093453), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %full_default_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%randn, 2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %mul_3), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_1, [1]), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %sum_1), kwargs = {})
triton_poi_fused_add_lift_fresh_log_mul_pow_sub_sum_1 = async_compile.triton('triton_poi_fused_add_lift_fresh_log_mul_pow_sub_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': 'i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_lift_fresh_log_mul_pow_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_lift_fresh_log_mul_pow_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = in_ptr0
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp9 = tl.load(in_ptr2 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp15 = tl.load(in_ptr1 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tl.load(in_ptr2 + (1))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp23 = tl.load(in_ptr1 + (2))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp25 = tl.load(in_ptr2 + (2))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp31 = tl.load(in_ptr1 + (3))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp33 = tl.load(in_ptr2 + (3))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp1 = tmp0.to(tl.float64)
    tmp2 = tl.full([1], -0.5, tl.float64)
    tmp3 = tmp2 * tmp1
    tmp4 = tl.full([1], 1.8378770664093453, tl.float64)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp11 = tmp10 * tmp10
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp8 + tmp13
    tmp19 = tmp18 * tmp18
    tmp20 = tmp19 * tmp12
    tmp21 = tmp16 + tmp20
    tmp22 = tmp14 + tmp21
    tmp27 = tmp26 * tmp26
    tmp28 = tmp27 * tmp12
    tmp29 = tmp24 + tmp28
    tmp30 = tmp22 + tmp29
    tmp35 = tmp34 * tmp34
    tmp36 = tmp35 * tmp12
    tmp37 = tmp32 + tmp36
    tmp38 = tmp30 + tmp37
    tmp39 = tmp6 - tmp38
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp39, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 4), (4, 1))
    assert_size_stride(primals_2, (1, 4), (4, 1))
    assert_size_stride(primals_3, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [eps], Original ATen: [aten.randn]
        buf0 = torch.ops.aten.randn.default([1, 4], dtype=torch.float32, device=device(type='cuda', index=0), pin_memory=False)
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((1, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [exp, mul, z], Original ATen: [aten.exp, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_exp_mul_0.run(primals_1, primals_2, buf1, buf2, 4, grid=grid(4), stream=stream0)
        del primals_1
        buf3 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_mul, wrapped_log, wrapped_mul_1, pow_1, mul_1, add_1, sum_1, log_p], Original ATen: [aten.lift_fresh, aten.mul, aten.log, aten.pow, aten.add, aten.sum, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_lift_fresh_log_mul_pow_sub_sum_1.run(primals_3.item(), primals_2, buf1, buf3, 1, grid=grid(1), stream=stream0)
        del primals_3
    return (buf2, buf3, primals_2, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((), (), device='cpu', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
