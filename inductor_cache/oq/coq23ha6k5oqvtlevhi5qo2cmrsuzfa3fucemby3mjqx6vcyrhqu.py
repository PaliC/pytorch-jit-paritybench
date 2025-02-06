# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/ob/cob2wqugnowlf7yik46mcu7qq2czarv6k3ma7r366pvtflfiy7qn.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.tanh]
# Source node to ATen node mapping:
#   input_2 => tanh
# Graph fragment:
#   %tanh : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%mm,), kwargs = {})
triton_poi_fused_tanh_0 = async_compile.triton('triton_poi_fused_tanh_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tanh_0(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = libdevice.tanh(tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dd/cddv4mzq32qgxwgzegif4jbv6g7zizygymnvjpn35osys7z4p7zf.py
# Topologically Sorted Source Nodes: [ws], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   ws => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%squeeze, [0], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%squeeze, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_1 = async_compile.triton('triton_poi_fused__softmax_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.load(in_ptr0 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (2))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (3))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp5 = triton_helpers.maximum(tmp2, tmp4)
    tmp8 = triton_helpers.maximum(tmp5, tmp7)
    tmp11 = triton_helpers.maximum(tmp8, tmp10)
    tmp12 = tmp0 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bj/cbj4osqfbcheu47wl6cty5dt2jtmi5dvkgvj3ogqy7q6nx5tmepz.py
# Topologically Sorted Source Nodes: [ws], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   ws => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [0], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_2 = async_compile.triton('triton_poi_fused__softmax_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.load(in_ptr0 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (2))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (3))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp5 = tmp2 + tmp4
    tmp8 = tmp5 + tmp7
    tmp11 = tmp8 + tmp10
    tmp12 = tmp0 / tmp11
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m2/cm23hlfvvno76pwyzvgs372bt5oigb3zmsghw4td3efcd6w7kwwt.py
# Topologically Sorted Source Nodes: [mul, agg_neib_1], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   agg_neib_1 => sum_2
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %unsqueeze), kwargs = {})
#   %sum_2 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (1))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp8 = tl.load(in_ptr1 + (2))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp12 = tl.load(in_ptr1 + (3))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp6 = tmp0 * tmp5
    tmp7 = tmp3 + tmp6
    tmp10 = tmp0 * tmp9
    tmp11 = tmp7 + tmp10
    tmp14 = tmp0 * tmp13
    tmp15 = tmp11 + tmp14
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ld/cldvjgpz4vabcesaclhqs2ttsizazy6pwgxu2e6tivc2y4ahcq64.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_1 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%cat,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_4 = async_compile.triton('triton_poi_fused_relu_threshold_backward_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (32, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (32, 32), (32, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, 4), (4, 1))
    assert_size_stride(primals_6, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.mm]
        extern_kernels.mm(primals_2, reinterpret_tensor(primals_1, (4, 32), (1, 4), 0), out=buf0)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_tanh_0.run(buf1, 128, grid=grid(128), stream=stream0)
        buf2 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf1, reinterpret_tensor(primals_3, (32, 32), (1, 32), 0), out=buf2)
        buf3 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.mm]
        extern_kernels.mm(primals_4, reinterpret_tensor(primals_1, (4, 32), (1, 4), 0), out=buf3)
        del primals_1
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_tanh_0.run(buf4, 128, grid=grid(128), stream=stream0)
        buf5 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_3, (32, 32), (1, 32), 0), out=buf5)
        buf6 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (4, 1, 32), (32, 32, 1), 0), reinterpret_tensor(buf5, (4, 32, 1), (32, 1, 1), 0), out=buf6)
        buf7 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [ws], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf6, buf7, 4, grid=grid(4), stream=stream0)
        buf8 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [ws], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf7, buf8, 4, grid=grid(4), stream=stream0)
        del buf7
        buf9 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, agg_neib_1], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sum_3.run(primals_2, buf8, buf9, 16, grid=grid(16), stream=stream0)
        del buf8
        buf12 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        buf10 = reinterpret_tensor(buf12, (4, 4), (8, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(primals_4, reinterpret_tensor(primals_5, (4, 4), (1, 4), 0), out=buf10)
        del primals_5
        buf11 = reinterpret_tensor(buf12, (4, 4), (8, 1), 4)  # alias
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf11)
        buf13 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        buf14 = empty_strided_cuda((4, 8), (8, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_4.run(buf12, buf13, buf14, 32, grid=grid(32), stream=stream0)
        del buf10
        del buf11
        del buf12
    return (buf13, primals_2, primals_4, buf1, buf4, buf6, buf9, buf14, primals_6, reinterpret_tensor(buf2, (4, 32, 1), (32, 1, 32), 0), reinterpret_tensor(buf5, (4, 1, 32), (32, 1, 1), 0), primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
