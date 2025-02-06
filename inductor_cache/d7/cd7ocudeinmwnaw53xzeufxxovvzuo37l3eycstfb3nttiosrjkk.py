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


# kernel path: inductor_cache/t4/ct4wkqchvwoefr3jqasznllepykyouhgfzbz4hdtsbduxytt5ui7.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   h => mul
# Graph fragment:
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %primals_2), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = in_ptr1
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u3/cu3tsfwe4mjt6hdvhixud4la7zqtzwakfr6pq7rzswm57jtpuiot.py
# Topologically Sorted Source Nodes: [h_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h_1 => convolution
# Graph fragment:
#   %convolution : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%mul, %primals_3, %primals_4, [1, 1], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 49) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5l/c5lw5rimlxllkf4ly3jbwmngba7uadjkouurj76f7ccbhc7llysm.py
# Topologically Sorted Source Nodes: [h_2, mul_1, mean, add, dom, h_3, h_4], Original ATen: [aten.leaky_relu, aten.mul, aten.mean, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#   add => add
#   dom => rsqrt
#   h_2 => gt, mul_1, where
#   h_3 => mul_3
#   h_4 => mul_4
#   mean => mean
#   mul_1 => mul_2
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %where), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_2, [1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-08), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %rsqrt), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %primals_5), kwargs = {})
triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_2 = async_compile.triton('triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': 'fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 49)
    x2 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp6 = tl.load(in_ptr0 + (x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (49 + x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (98 + x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (147 + x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp35 = in_ptr1
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp6 > tmp1
    tmp8 = tmp6 * tmp3
    tmp9 = tl.where(tmp7, tmp6, tmp8)
    tmp10 = tmp9 * tmp9
    tmp12 = tmp11 > tmp1
    tmp13 = tmp11 * tmp3
    tmp14 = tl.where(tmp12, tmp11, tmp13)
    tmp15 = tmp14 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp17 > tmp1
    tmp19 = tmp17 * tmp3
    tmp20 = tl.where(tmp18, tmp17, tmp19)
    tmp21 = tmp20 * tmp20
    tmp22 = tmp16 + tmp21
    tmp24 = tmp23 > tmp1
    tmp25 = tmp23 * tmp3
    tmp26 = tl.where(tmp24, tmp23, tmp25)
    tmp27 = tmp26 * tmp26
    tmp28 = tmp22 + tmp27
    tmp29 = 4.0
    tmp30 = tmp28 / tmp29
    tmp31 = 1e-08
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp5 * tmp33
    tmp36 = tmp34 * tmp35
    tl.store(in_out_ptr0 + (x3), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6h/c6hrfquxshsv5k2iyjwwhqcyp2hfm3m2kufkxt6kbgpioujtkhv3.py
# Topologically Sorted Source Nodes: [h_6, mul_4, mean_2, add_1, dom_1, h_7], Original ATen: [aten.leaky_relu, aten.mul, aten.mean, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#   add_1 => add_1
#   dom_1 => rsqrt_1
#   h_6 => gt_1, mul_5, where_1
#   h_7 => mul_7
#   mean_2 => mean_1
#   mul_4 => mul_6
# Graph fragment:
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.2), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_1, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %where_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_6, [1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-08), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %rsqrt_1), kwargs = {})
triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_3 = async_compile.triton('triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 49)
    x2 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp6 = tl.load(in_ptr0 + (x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (49 + x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (98 + x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (147 + x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp6 > tmp1
    tmp8 = tmp6 * tmp3
    tmp9 = tl.where(tmp7, tmp6, tmp8)
    tmp10 = tmp9 * tmp9
    tmp12 = tmp11 > tmp1
    tmp13 = tmp11 * tmp3
    tmp14 = tl.where(tmp12, tmp11, tmp13)
    tmp15 = tmp14 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp17 > tmp1
    tmp19 = tmp17 * tmp3
    tmp20 = tl.where(tmp18, tmp17, tmp19)
    tmp21 = tmp20 * tmp20
    tmp22 = tmp16 + tmp21
    tmp24 = tmp23 > tmp1
    tmp25 = tmp23 * tmp3
    tmp26 = tl.where(tmp24, tmp23, tmp25)
    tmp27 = tmp26 * tmp26
    tmp28 = tmp22 + tmp27
    tmp29 = 4.0
    tmp30 = tmp28 / tmp29
    tmp31 = 1e-08
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp5 * tmp33
    tl.store(out_ptr0 + (x3), tmp34, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (), ())
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (), ())
    assert_size_stride(primals_6, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(primals_1, primals_2.item(), buf0, 256, grid=grid(256), stream=stream0)
        del primals_1
        del primals_2
        # Topologically Sorted Source Nodes: [h_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_3, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 7, 7), (196, 49, 7, 1))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [h_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf2, primals_4, 784, grid=grid(784), stream=stream0)
        del primals_4
        buf3 = empty_strided_cuda((4, 4, 7, 7), (196, 49, 7, 1), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [h_2, mul_1, mean, add, dom, h_3, h_4], Original ATen: [aten.leaky_relu, aten.mul, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_2.run(buf4, buf2, primals_5.item(), 784, grid=grid(784), stream=stream0)
        # Topologically Sorted Source Nodes: [h_5], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 4, 7, 7), (196, 49, 7, 1))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [h_5], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf6, primals_7, 784, grid=grid(784), stream=stream0)
        del primals_7
        buf7 = empty_strided_cuda((4, 4, 7, 7), (196, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_6, mul_4, mean_2, add_1, dom_1, h_7], Original ATen: [aten.leaky_relu, aten.mul, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_3.run(buf6, buf7, 784, grid=grid(784), stream=stream0)
    return (buf7, primals_3, primals_5, primals_6, buf0, buf2, buf4, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
