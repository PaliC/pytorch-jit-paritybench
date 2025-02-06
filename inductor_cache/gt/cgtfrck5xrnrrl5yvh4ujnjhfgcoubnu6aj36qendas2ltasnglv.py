# AOT ID: ['25_forward']
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


# kernel path: inductor_cache/nz/cnzorawtmfyihuacz4ddlc3p6xkrno6bz2clbmhkc6ypmnrsmth7.py
# Topologically Sorted Source Nodes: [exp, add, log, abs_1, beta, dz, r, add_1, h_arr, neg, mul, pow_1, h_arr_, sub_2, add_4, log_1, mul_2, add_6, log_2, log_det], Original ATen: [aten.exp, aten.add, aten.log, aten.abs, aten.sub, aten.linalg_vector_norm, aten.div, aten.neg, aten.mul, aten.pow]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   add => add
#   add_1 => add_1
#   add_4 => add_4
#   add_6 => add_6
#   beta => sub
#   dz => sub_1
#   exp => exp
#   h_arr => div
#   h_arr_ => div_1
#   log => log
#   log_1 => log_1
#   log_2 => log_2
#   log_det => add_7
#   mul => mul
#   mul_2 => mul_2
#   neg => neg
#   pow_1 => pow_3
#   r => pow_1, pow_2, sum_1
#   sub_2 => sub_2
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%primals_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
#   %abs_1 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%primals_2,), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%log, %abs_1), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_3), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, %pow_2), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %add_1), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sub,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %pow_2), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1, 2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %pow_3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_5, 1), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, 1), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_4,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %log_1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %div_1), kwargs = {})
#   %log_2 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_6,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %log_2), kwargs = {})
triton_poi_fused_abs_add_div_exp_linalg_vector_norm_log_mul_neg_pow_sub_0 = async_compile.triton('triton_poi_fused_abs_add_div_exp_linalg_vector_norm_log_mul_neg_pow_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_exp_linalg_vector_norm_log_mul_neg_pow_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_div_exp_linalg_vector_norm_log_mul_neg_pow_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16
    x4 = (xindex % 16)
    x0 = (xindex % 4)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x4 + 64*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (16 + x4 + 64*x2), xmask)
    tmp8 = tl.load(in_ptr0 + (32 + x4 + 64*x2), xmask)
    tmp12 = tl.load(in_ptr0 + (48 + x4 + 64*x2), xmask)
    tmp17 = tl.load(in_ptr2 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp22 = tl.load(in_ptr3 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp28 = tl.load(in_ptr4 + (0))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp5 = tmp4 - tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8 - tmp1
    tmp10 = tmp9 * tmp9
    tmp11 = tmp7 + tmp10
    tmp13 = tmp12 - tmp1
    tmp14 = tmp13 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp19 = tl.full([1], 1, tl.int64)
    tmp20 = tmp18 - tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp24 = tl_math.exp(tmp23)
    tmp25 = 1.0
    tmp26 = tmp24 + tmp25
    tmp27 = tl_math.log(tmp26)
    tmp30 = tl_math.abs(tmp29)
    tmp31 = tmp27 - tmp30
    tmp32 = tmp30 + tmp16
    tmp33 = tmp31 / tmp32
    tmp34 = tmp33 + tmp25
    tmp35 = tl_math.log(tmp34)
    tmp36 = tmp21 * tmp35
    tmp37 = -tmp31
    tmp38 = tmp37 * tmp16
    tmp39 = tmp32 * tmp32
    tmp40 = tmp38 / tmp39
    tmp41 = tmp34 + tmp40
    tmp42 = tl_math.log(tmp41)
    tmp43 = tmp36 + tmp42
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k5/ck54mogighltmkrknqfrmp2oy7mcxx7wd3psruo6rhae64lupuzq.py
# Topologically Sorted Source Nodes: [exp, add, log, abs_1, beta, dz, add_1, h_arr, mul_1, z_], Original ATen: [aten.exp, aten.add, aten.log, aten.abs, aten.sub, aten.div, aten.mul]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   add => add
#   add_1 => add_1
#   beta => sub
#   dz => sub_1
#   exp => exp
#   h_arr => div
#   log => log
#   mul_1 => mul_1
#   z_ => add_3
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%primals_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
#   %abs_1 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%primals_2,), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%log, %abs_1), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_3), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, %pow_2), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %add_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sub_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, %mul_1), kwargs = {})
triton_poi_fused_abs_add_div_exp_log_mul_sub_1 = async_compile.triton('triton_poi_fused_abs_add_div_exp_log_mul_sub_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_exp_log_mul_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_div_exp_log_mul_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x3 = xindex // 64
    x5 = (xindex % 16)
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (x5 + 16*x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl_math.exp(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tl_math.log(tmp5)
    tmp9 = tl_math.abs(tmp8)
    tmp10 = tmp6 - tmp9
    tmp12 = tmp9 + tmp11
    tmp13 = tmp10 / tmp12
    tmp15 = tmp0 - tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (1, ), (1, ))
    assert_size_stride(primals_2, (1, ), (1, ))
    assert_size_stride(primals_3, (1, 4), (4, 1))
    assert_size_stride(primals_4, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_5, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        buf2 = empty_strided_cuda((4, 1, 4, 4), (16, 1, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [exp, add, log, abs_1, beta, dz, r, add_1, h_arr, neg, mul, pow_1, h_arr_, sub_2, add_4, log_1, mul_2, add_6, log_2, log_det], Original ATen: [aten.exp, aten.add, aten.log, aten.abs, aten.sub, aten.linalg_vector_norm, aten.div, aten.neg, aten.mul, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_div_exp_linalg_vector_norm_log_mul_neg_pow_sub_0.run(primals_4, primals_3, primals_5, primals_1, primals_2, buf0, buf2, 64, grid=grid(64), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [exp, add, log, abs_1, beta, dz, add_1, h_arr, mul_1, z_], Original ATen: [aten.exp, aten.add, aten.log, aten.abs, aten.sub, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_div_exp_log_mul_sub_1.run(primals_4, primals_1, primals_2, buf0, primals_3, buf1, 256, grid=grid(256), stream=stream0)
        del buf0
    return (buf1, reinterpret_tensor(buf2, (64, ), (1, ), 0), primals_1, primals_2, primals_3, primals_4, primals_5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
