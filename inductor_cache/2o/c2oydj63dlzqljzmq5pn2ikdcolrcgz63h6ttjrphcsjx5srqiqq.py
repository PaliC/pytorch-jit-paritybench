# AOT ID: ['16_forward']
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


# kernel path: inductor_cache/og/cogyqond72szqf7zl47o6x3kiufx4kqx3pjgbadtmqp6qe46mwoa.py
# Topologically Sorted Source Nodes: [input_2, sigmoid, input_3, weight, mul_1], Original ATen: [aten.repeat, aten._native_batch_norm_legit, aten.sigmoid, aten.mul, aten.exp]
# Source node to ATen node mapping:
#   input_2 => add, repeat, repeat_1, rsqrt, var_mean
#   input_3 => mul_2
#   mul_1 => mul_3
#   sigmoid => sigmoid
#   weight => exp
# Graph fragment:
#   %repeat : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_3, [4]), kwargs = {})
#   %repeat_1 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_4, [4]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, 12.0), kwargs = {})
#   %exp : [num_users=3] = call_function[target=torch.ops.aten.exp.default](args = (%mul_2,), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %exp), kwargs = {})
triton_per_fused__native_batch_norm_legit_exp_mul_repeat_sigmoid_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_exp_mul_repeat_sigmoid_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_exp_mul_repeat_sigmoid_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_exp_mul_repeat_sigmoid_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + ((x0 % 4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((x0 % 4)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (r1 + 16*x0), xmask, other=0.0)
    tmp32 = tl.load(in_ptr3 + (r1 + 16*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 16.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp2 - tmp12
    tmp25 = tmp24 * tmp23
    tmp26 = tmp25 * tmp0
    tmp27 = tmp26 + tmp1
    tmp28 = tl.sigmoid(tmp27)
    tmp29 = 12.0
    tmp30 = tmp28 * tmp29
    tmp31 = tl_math.exp(tmp30)
    tmp33 = tmp32 * tmp31
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp1, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr3 + (r1 + 16*x0), tmp31, xmask)
    tl.store(out_ptr4 + (r1 + 16*x0), tmp33, xmask)
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hu/chuqta4pwag2sed3mevuh7cp562a3myjskaclwmmtmn4ighmgjye.py
# Topologically Sorted Source Nodes: [avg_pool2d, avg_pool2d_1, frac], Original ATen: [aten.avg_pool2d, aten.div]
# Source node to ATen node mapping:
#   avg_pool2d => avg_pool2d
#   avg_pool2d_1 => avg_pool2d_1
#   frac => div
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_3, [3, 3], [2, 2], [1, 1]), kwargs = {})
#   %avg_pool2d_1 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%exp, [3, 3], [2, 2], [1, 1]), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%avg_pool2d, %avg_pool2d_1), kwargs = {})
triton_poi_fused_avg_pool2d_div_1 = async_compile.triton('triton_poi_fused_avg_pool2d_div_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_div_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_div_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x4 = xindex // 2
    x3 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + 2*x0 + 8*x4), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + 2*x0 + 8*x4), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + 2*x0 + 8*x4), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 8*x4), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 8*x4), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x4), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3 + 2*x0 + 8*x4), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x4), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x4), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5)))*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5))) + ((-2)*x0*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5)))) + ((-2)*x1*((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5)))) + 4*x0*x1 + ((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5))) + ((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5)))
    tmp53 = tmp51 / tmp52
    tmp54 = tl.load(in_ptr1 + ((-5) + 2*x0 + 8*x4), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tl.load(in_ptr1 + ((-4) + 2*x0 + 8*x4), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp55 + tmp54
    tmp57 = tl.load(in_ptr1 + ((-3) + 2*x0 + 8*x4), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tmp57 + tmp56
    tmp59 = tl.load(in_ptr1 + ((-1) + 2*x0 + 8*x4), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tmp59 + tmp58
    tmp61 = tl.load(in_ptr1 + (2*x0 + 8*x4), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp61 + tmp60
    tmp63 = tl.load(in_ptr1 + (1 + 2*x0 + 8*x4), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp63 + tmp62
    tmp65 = tl.load(in_ptr1 + (3 + 2*x0 + 8*x4), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp65 + tmp64
    tmp67 = tl.load(in_ptr1 + (4 + 2*x0 + 8*x4), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp67 + tmp66
    tmp69 = tl.load(in_ptr1 + (5 + 2*x0 + 8*x4), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp69 + tmp68
    tmp71 = tmp70 / tmp52
    tmp72 = tmp53 / tmp71
    tl.store(out_ptr0 + (x3), tmp53, xmask)
    tl.store(out_ptr1 + (x3), tmp71, xmask)
    tl.store(out_ptr2 + (x3), tmp72, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 4, 4), (64, 16, 4, 1))
        buf1 = empty_strided_cuda((16, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((16, ), (1, ), torch.float32)
        buf3 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf6 = reinterpret_tensor(buf4, (1, 16, 1, 1), (16, 1, 1, 1), 0); del buf4  # reuse
        buf7 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, sigmoid, input_3, weight, mul_1], Original ATen: [aten.repeat, aten._native_batch_norm_legit, aten.sigmoid, aten.mul, aten.exp]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_exp_mul_repeat_sigmoid_0.run(buf6, primals_3, primals_4, buf0, primals_2, buf1, buf2, buf3, buf7, buf8, 16, 16, grid=grid(16), stream=stream0)
        del primals_3
        del primals_4
        buf9 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf10 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf11 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [avg_pool2d, avg_pool2d_1, frac], Original ATen: [aten.avg_pool2d, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_div_1.run(buf8, buf7, buf9, buf10, buf11, 64, grid=grid(64), stream=stream0)
    return (buf11, primals_1, primals_2, buf0, buf1, buf2, buf3, buf6, buf7, buf8, buf9, buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
