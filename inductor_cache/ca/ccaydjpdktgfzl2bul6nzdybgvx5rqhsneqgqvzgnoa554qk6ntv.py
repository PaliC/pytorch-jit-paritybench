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


# kernel path: inductor_cache/ht/chtunlhl2bp6atayhcivpzj6angmh7fs2pk2ly3pasprn7lxye2v.py
# Topologically Sorted Source Nodes: [conv1d, clamp, norm_att], Original ATen: [aten.convolution, aten.clamp, aten._softmax]
# Source node to ATen node mapping:
#   clamp => clamp_max, clamp_min
#   conv1d => convolution
#   norm_att => amax, exp, sub, sum_1
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1], [0], [1], False, [0], 1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution, -10), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 10), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clamp_max, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
triton_poi_fused__softmax_clamp_convolution_0 = async_compile.triton('triton_poi_fused__softmax_clamp_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_clamp_convolution_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_clamp_convolution_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -10.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 10.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp8 = tmp7 + tmp1
    tmp9 = triton_helpers.maximum(tmp8, tmp3)
    tmp10 = triton_helpers.minimum(tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp6, tmp10)
    tmp13 = tmp12 + tmp1
    tmp14 = triton_helpers.maximum(tmp13, tmp3)
    tmp15 = triton_helpers.minimum(tmp14, tmp5)
    tmp16 = triton_helpers.maximum(tmp11, tmp15)
    tmp18 = tmp17 + tmp1
    tmp19 = triton_helpers.maximum(tmp18, tmp3)
    tmp20 = triton_helpers.minimum(tmp19, tmp5)
    tmp21 = triton_helpers.maximum(tmp16, tmp20)
    tmp22 = tmp6 - tmp21
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp10 - tmp21
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tmp23 + tmp25
    tmp27 = tmp15 - tmp21
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp20 - tmp21
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr0 + (x2), tmp21, xmask)
    tl.store(out_ptr1 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bo/cbom3t5z4ks5qhze7k75avltmqtqpowtdkhblsohkbzmhsqyzv3g.py
# Topologically Sorted Source Nodes: [conv1d, clamp, norm_att], Original ATen: [aten.convolution, aten.clamp, aten._softmax, aten.ge, aten.le, aten.logical_and]
# Source node to ATen node mapping:
#   clamp => clamp_max, clamp_min
#   conv1d => convolution
#   norm_att => amax, div, exp, sub
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1], [0], [1], False, [0], 1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution, -10), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 10), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clamp_max, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution, -10), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution, 10), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %le), kwargs = {})
triton_poi_fused__softmax_clamp_convolution_ge_le_logical_and_1 = async_compile.triton('triton_poi_fused__softmax_clamp_convolution_ge_le_logical_and_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_clamp_convolution_ge_le_logical_and_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_clamp_convolution_ge_le_logical_and_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    x4 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -10.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 10.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp8 = tmp6 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp11 = tmp9 / tmp10
    tmp12 = tmp2 >= tmp3
    tmp13 = tmp2 <= tmp5
    tmp14 = tmp12 & tmp13
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jj/cjjgzvniqz7odbf3lar234k3k33qlpbnpwwt3vanina36rqkuozh.py
# Topologically Sorted Source Nodes: [cla], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   cla => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_4, %primals_5, [1], [0], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fj/cfjlbssbk4ais2qzzbmvxvqhmwj5bwxmy5diqi7avyianeiddw4e.py
# Topologically Sorted Source Nodes: [mul, x], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul => mul
#   x => sum_2
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %convolution_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [2]), kwargs = {})
triton_poi_fused_mul_sum_3 = async_compile.triton('triton_poi_fused_mul_sum_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 1), (4, 1, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_4, (4, 4, 1), (4, 1, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 4), (16, 4, 1))
        buf1 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf2 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [conv1d, clamp, norm_att], Original ATen: [aten.convolution, aten.clamp, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_clamp_convolution_0.run(buf0, primals_2, buf1, buf2, 16, grid=grid(16), stream=stream0)
        buf3 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv1d, clamp, norm_att], Original ATen: [aten.convolution, aten.clamp, aten._softmax, aten.ge, aten.le, aten.logical_and]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_clamp_convolution_ge_le_logical_and_1.run(buf0, primals_2, buf1, buf2, buf3, buf7, 64, grid=grid(64), stream=stream0)
        del buf0
        del buf1
        del primals_2
        # Topologically Sorted Source Nodes: [cla], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(primals_3, primals_4, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf4, (4, 4, 4), (16, 4, 1))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [cla], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf5, primals_5, 64, grid=grid(64), stream=stream0)
        del primals_5
        buf6 = reinterpret_tensor(buf2, (4, 4), (4, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [mul, x], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sum_3.run(buf3, buf5, buf6, 16, grid=grid(16), stream=stream0)
    return (buf6, buf3, buf5, primals_1, primals_3, primals_4, buf3, buf5, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
