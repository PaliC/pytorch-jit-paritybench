# AOT ID: ['13_forward']
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


# kernel path: inductor_cache/s6/cs6psy7cpta5s4p3aiyergpro2heioybe5evhugzwvd2o2ncbu3d.py
# Topologically Sorted Source Nodes: [lt, mask, masked_fm, sum_1, sum_2, add, margin], Original ATen: [aten.lt, aten._to_copy, aten.mul, aten.sum, aten.add, aten.div]
# Source node to ATen node mapping:
#   add => add
#   lt => lt
#   margin => div
#   mask => convert_element_type
#   masked_fm => mul
#   sum_1 => sum_1
#   sum_2 => sum_2
# Graph fragment:
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%primals_1, 0.0), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %convert_element_type), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [0, 2, 3], True), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type, [0, 2, 3], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, 1e-06), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %add), kwargs = {})
triton_per_fused__to_copy_add_div_lt_mul_sum_0 = async_compile.triton('triton_per_fused__to_copy_add_div_lt_mul_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_lt_mul_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_div_lt_mul_sum_0(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = (rindex % 16)
    r2 = rindex // 16
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0 + 64*r2), xmask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 < tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = tmp8 / tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dy/cdy6v7jexh3hbtc7h2ed6usfvzholsrnixtiphpsoqhgaeakducz.py
# Topologically Sorted Source Nodes: [fm_t, input_2, le, le_1, and_, float_2, mask_1, sub_1, pow_1, mul_1, loss], Original ATen: [aten.maximum, aten._native_batch_norm_legit_no_training, aten.le, aten.bitwise_and, aten._to_copy, aten.rsub, aten.sub, aten.pow, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   float_2 => convert_element_type_3
#   fm_t => maximum
#   input_2 => add_2, mul_2, mul_3, sub
#   le => le
#   le_1 => le_1
#   loss => mean
#   mask_1 => sub_1
#   mul_1 => mul_4
#   pow_1 => pow_1
#   sub_1 => sub_2
# Graph fragment:
#   %maximum : [num_users=3] = call_function[target=torch.ops.aten.maximum.default](args = (%primals_1, %div), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %unsqueeze_5), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %unsqueeze_7), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_2, %maximum), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%maximum, 0.0), kwargs = {})
#   %bitwise_and : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%le, %le_1), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type_3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %maximum), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, %sub_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_4,), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training__to_copy_bitwise_and_le_maximum_mean_mul_pow_rsub_sub_1 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training__to_copy_bitwise_and_le_maximum_mean_mul_pow_rsub_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training__to_copy_bitwise_and_le_maximum_mean_mul_pow_rsub_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training__to_copy_bitwise_and_le_maximum_mean_mul_pow_rsub_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r3 = rindex
    r1 = ((rindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (r3), None)
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (r3), None)
    tmp17 = tl.load(in_ptr6 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = tmp15 <= tmp18
    tmp20 = 0.0
    tmp21 = tmp18 <= tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp15 - tmp18
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22.to(tl.float32)
    tmp26 = tmp9 - tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = 256.0
    tmp32 = tmp30 / tmp31
    tl.store(out_ptr1 + (tl.broadcast_to(r3, [RBLOCK])), tmp22, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        buf2 = reinterpret_tensor(buf0, (1, 4, 1, 1), (4, 1, 1, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [lt, mask, masked_fm, sum_1, sum_2, add, margin], Original ATen: [aten.lt, aten._to_copy, aten.mul, aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_div_lt_mul_sum_0.run(buf2, primals_1, 4, 64, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_3, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 4, 4), (64, 16, 4, 1))
        buf5 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [fm_t, input_2, le, le_1, and_, float_2, mask_1, sub_1, pow_1, mul_1, loss], Original ATen: [aten.maximum, aten._native_batch_norm_legit_no_training, aten.le, aten.bitwise_and, aten._to_copy, aten.rsub, aten.sub, aten.pow, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training__to_copy_bitwise_and_le_maximum_mean_mul_pow_rsub_sub_1.run(buf7, buf3, primals_4, primals_5, primals_6, primals_7, primals_1, buf2, buf5, 1, 256, grid=grid(1), stream=stream0)
    return (buf7, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, buf2, buf3, buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
