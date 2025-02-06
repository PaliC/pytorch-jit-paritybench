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


# kernel path: inductor_cache/az/caz75y5sjaozyyd6znbmwd6ijpiaegvkbafun42wq3de23bssxpd.py
# Topologically Sorted Source Nodes: [out1, out2, out3, out4, out1_1, out2_1, out3_1, out4_1, add, add_1, out], Original ATen: [aten.convolution, aten.elu, aten.add]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   out => add_2
#   out1 => convolution
#   out1_1 => expm1, gt, mul, mul_2, where
#   out2 => convolution_1
#   out2_1 => expm1_1, gt_1, mul_3, mul_5, where_1
#   out3 => convolution_2
#   out3_1 => expm1_2, gt_2, mul_6, mul_8, where_2
#   out4 => convolution_3
#   out4_1 => expm1_3, gt_3, mul_11, mul_9, where_3
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_4, %primals_5, [1, 1], [2, 2], [2, 2], False, [0, 0], 1), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_6, %primals_7, [1, 1], [3, 3], [3, 3], False, [0, 0], 1), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_8, %primals_9, [1, 1], [4, 4], [4, 4], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul, %mul_2), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 1.0), kwargs = {})
#   %expm1_1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_3,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_1, 1.0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %mul_3, %mul_5), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, 1.0), kwargs = {})
#   %expm1_2 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_6,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_2, 1.0), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %mul_6, %mul_8), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_3, 0), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_3, 1.0), kwargs = {})
#   %expm1_3 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_9,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_3, 1.0), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %mul_9, %mul_11), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %where_2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %where_3), kwargs = {})
triton_poi_fused_add_convolution_elu_0 = async_compile.triton('triton_poi_fused_add_convolution_elu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_elu_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_elu_0(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr3 + (x3), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = 0.0
    tmp13 = tmp2 > tmp12
    tmp14 = 1.0
    tmp15 = tmp2 * tmp14
    tmp16 = libdevice.expm1(tmp15)
    tmp17 = tmp16 * tmp14
    tmp18 = tl.where(tmp13, tmp15, tmp17)
    tmp19 = tmp5 > tmp12
    tmp20 = tmp5 * tmp14
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp14
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tmp24 = tmp18 + tmp23
    tmp25 = tmp8 > tmp12
    tmp26 = tmp8 * tmp14
    tmp27 = libdevice.expm1(tmp26)
    tmp28 = tmp27 * tmp14
    tmp29 = tl.where(tmp25, tmp26, tmp28)
    tmp30 = tmp24 + tmp29
    tmp31 = tmp11 > tmp12
    tmp32 = tmp11 * tmp14
    tmp33 = libdevice.expm1(tmp32)
    tmp34 = tmp33 * tmp14
    tmp35 = tl.where(tmp31, tmp32, tmp34)
    tmp36 = tmp30 + tmp35
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(in_out_ptr1 + (x3), tmp5, xmask)
    tl.store(in_out_ptr2 + (x3), tmp8, xmask)
    tl.store(in_out_ptr3 + (x3), tmp11, xmask)
    tl.store(out_ptr0 + (x3), tmp36, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out2], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(primals_3, primals_4, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out3], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(primals_3, primals_6, stride=(1, 1), padding=(3, 3), dilation=(3, 3), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out4], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(primals_3, primals_8, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 4, 4, 4), (64, 16, 4, 1))
        buf1 = buf0; del buf0  # reuse
        buf3 = buf2; del buf2  # reuse
        buf5 = buf4; del buf4  # reuse
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1, out2, out3, out4, out1_1, out2_1, out3_1, out4_1, add, add_1, out], Original ATen: [aten.convolution, aten.elu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_elu_0.run(buf1, buf3, buf5, buf7, primals_2, primals_5, primals_7, primals_9, buf8, 256, grid=grid(256), stream=stream0)
        del primals_2
        del primals_5
        del primals_7
        del primals_9
    return (buf8, primals_1, primals_3, primals_4, primals_6, primals_8, buf1, buf3, buf5, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
