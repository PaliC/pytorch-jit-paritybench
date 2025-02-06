# AOT ID: ['166_forward']
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


# kernel path: inductor_cache/df/cdflsf5lb4dyk2cb4c6xxyxxy4mxv2gxpjsdvo5umyoxvltw5sep.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_5, input_6, value, y], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => relu_1
#   value => add_4
#   y => mean
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, %relu_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_4, [-1, -2], True), kwargs = {})
triton_red_fused__native_batch_norm_legit_no_training_add_mean_relu_0 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_add_mean_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_add_mean_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_add_mean_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = (xindex % 32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + 4096*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr5 + (r2 + 4096*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1, 1], 1, tl.int32)
        tmp8 = tmp7 / tmp6
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full([1, 1], 0, tl.int32)
        tmp17 = triton_helpers.maximum(tmp16, tmp15)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp21 + tmp4
        tmp23 = libdevice.sqrt(tmp22)
        tmp24 = tmp7 / tmp23
        tmp25 = tmp24 * tmp9
        tmp26 = tmp20 * tmp25
        tmp28 = tmp26 * tmp27
        tmp30 = tmp28 + tmp29
        tmp31 = triton_helpers.maximum(tmp16, tmp30)
        tmp32 = tmp17 + tmp31
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
        tl.store(out_ptr0 + (r2 + 4096*x3), tmp32, rmask & xmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tmp36 = 4096.0
    tmp37 = tmp34 / tmp36
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp37, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7x/c7xf5fqwt2wp2a3qcxbmmx4zl72xg5zos6ajxfszjfsu333y5kmv.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul_6
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %expand), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_poi_fused_mul_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (4, 32, 64, 64), (131072, 4096, 64, 1))
    assert_size_stride(primals_2, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, ), (1, ))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (1, 1, 3), (3, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf0, (4, 32, 64, 64), (131072, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_1, primals_7, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf1, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf2 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3, input_5, input_6, value, y], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training_add_mean_relu_0.run(buf4, buf0, primals_3, primals_4, primals_5, primals_6, buf1, primals_8, primals_9, primals_10, primals_11, buf2, 128, 4096, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(reinterpret_tensor(buf4, (4, 1, 32), (32, 0, 1), 0), primals_12, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf5, (4, 1, 32), (32, 32, 1))
        buf6 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(buf6, buf5, 524288, grid=grid(524288), stream=stream0)
    return (buf6, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, buf0, buf1, reinterpret_tensor(buf4, (4, 1, 32), (32, 1, 1), 0), buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 32, 64, 64), (131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, 1, 3), (3, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
