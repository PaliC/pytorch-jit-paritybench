# AOT ID: ['11_forward']
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


# kernel path: inductor_cache/rx/crxu5j5sj7c2keb66mpeaow6u2xijqnu72j3oslsklqw4yhsgbor.py
# Topologically Sorted Source Nodes: [mul, exp, loc, mul_2, exp_1, log_scale, exp_2, mul_4, z], Original ATen: [aten.mul, aten.exp, aten.add]
# Source node to ATen node mapping:
#   exp => exp
#   exp_1 => exp_1
#   exp_2 => exp_2
#   loc => mul_1
#   log_scale => mul_3
#   mul => mul
#   mul_2 => mul_2
#   mul_4 => mul_4
#   z => add
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, 3.0), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %exp), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, 3.0), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_2,), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, %exp_1), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_3,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_2, %randn), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_4), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_mul_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = 3.0
    tmp3 = tmp1 * tmp2
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tmp8 = tmp7 * tmp2
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tmp6 * tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp5 + tmp13
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/it/citaggqcgbsahzd5ayopju4kxemzevd7nkrcpkox4zsx3ivoldd2.py
# Topologically Sorted Source Nodes: [mul_2, exp_1, log_scale, wrapped_mul, wrapped_log, wrapped_mul_1, sum_1, mul_5, sub, pow_1, sum_2, mul_6, log_p], Original ATen: [aten.mul, aten.exp, aten.lift_fresh, aten.log, aten.sum, aten.sub, aten.pow]
# Source node to ATen node mapping:
#   exp_1 => exp_1
#   log_p => sub_1
#   log_scale => mul_3
#   mul_2 => mul_2
#   mul_5 => mul_7
#   mul_6 => mul_8
#   pow_1 => pow_1
#   sub => sub
#   sum_1 => sum_1
#   sum_2 => sum_2
#   wrapped_log => full_default_1
#   wrapped_mul => full_default, mul_5
#   wrapped_mul_1 => mul_6
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, 3.0), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_2,), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, %exp_1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -0.5), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %primals_5), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.8378770664093453), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %full_default_1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [1]), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, %sum_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%randn, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, 0.5), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %mul_8), kwargs = {})
triton_poi_fused_exp_lift_fresh_log_mul_pow_sub_sum_1 = async_compile.triton('triton_poi_fused_exp_lift_fresh_log_mul_pow_sub_sum_1', '''
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
    triton_meta={'signature': {'in_ptr0': 'i64', 'in_ptr1': 'fp64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (2, 3, 4, 5), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_lift_fresh_log_mul_pow_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_exp_lift_fresh_log_mul_pow_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = in_ptr0
    tmp7 = in_ptr1
    tmp9 = tl.load(in_ptr2 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp17 = tl.load(in_ptr2 + (1))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp19 = tl.load(in_ptr3 + (1))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp25 = tl.load(in_ptr2 + (2))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp27 = tl.load(in_ptr3 + (2))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp33 = tl.load(in_ptr2 + (3))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp35 = tl.load(in_ptr3 + (3))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp43 = tl.load(in_ptr4 + (0))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp46 = tl.load(in_ptr4 + (1))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp50 = tl.load(in_ptr4 + (2))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp54 = tl.load(in_ptr4 + (3))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp1 = tmp0.to(tl.float64)
    tmp2 = tl.full([1], -0.5, tl.float64)
    tmp3 = tmp2 * tmp1
    tmp4 = tl.full([1], 1.8378770664093453, tl.float64)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp13 = 3.0
    tmp14 = tmp12 * tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp10 * tmp15
    tmp21 = tmp20 * tmp13
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp18 * tmp22
    tmp24 = tmp16 + tmp23
    tmp29 = tmp28 * tmp13
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp26 * tmp30
    tmp32 = tmp24 + tmp31
    tmp37 = tmp36 * tmp13
    tmp38 = tl_math.exp(tmp37)
    tmp39 = tmp34 * tmp38
    tmp40 = tmp32 + tmp39
    tmp41 = tmp8 * tmp40
    tmp42 = tmp6 - tmp41
    tmp45 = tmp44 * tmp44
    tmp48 = tmp47 * tmp47
    tmp49 = tmp45 + tmp48
    tmp52 = tmp51 * tmp51
    tmp53 = tmp49 + tmp52
    tmp56 = tmp55 * tmp55
    tmp57 = tmp53 + tmp56
    tmp58 = 0.5
    tmp59 = tmp57 * tmp58
    tmp60 = tmp42 - tmp59
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp60, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (1, 4), (4, 1))
    assert_size_stride(primals_2, (1, 4), (4, 1))
    assert_size_stride(primals_3, (1, 4), (4, 1))
    assert_size_stride(primals_4, (1, 4), (4, 1))
    assert_size_stride(primals_5, (), ())
    assert_size_stride(primals_6, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [eps], Original ATen: [aten.randn]
        buf0 = torch.ops.aten.randn.default([1, 4], dtype=torch.float32, device=device(type='cuda', index=0), pin_memory=False)
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((1, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, exp, loc, mul_2, exp_1, log_scale, exp_2, mul_4, z], Original ATen: [aten.mul, aten.exp, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_exp_mul_0.run(primals_1, primals_2, primals_3, primals_4, buf1, buf2, 4, grid=grid(4), stream=stream0)
        buf3 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul_2, exp_1, log_scale, wrapped_mul, wrapped_log, wrapped_mul_1, sum_1, mul_5, sub, pow_1, sum_2, mul_6, log_p], Original ATen: [aten.mul, aten.exp, aten.lift_fresh, aten.log, aten.sum, aten.sub, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_lift_fresh_log_mul_pow_sub_sum_1.run(primals_5.item(), primals_6.item(), primals_3, primals_4, buf1, buf3, 1, grid=grid(1), stream=stream0)
        del primals_5
    return (buf2, buf3, primals_1, primals_2, primals_3, primals_4, primals_6, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_6 = rand_strided((), (), device='cpu', dtype=torch.float64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
