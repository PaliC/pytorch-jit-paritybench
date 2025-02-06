# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/4s/c4slx4mpdpcbz2u6yv4emep5o7naictfjawcli7sqksxtvwrmcfx.py
# Topologically Sorted Source Nodes: [linear, x], Original ATen: [aten.addmm, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   linear => add_tensor_4
#   x => relu
# Graph fragment:
#   %add_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_4, %primals_2), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_4,), kwargs = {})
#   %le_4 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_addmm_relu_threshold_backward_0 = async_compile.triton('triton_poi_fused_addmm_relu_threshold_backward_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_threshold_backward_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_threshold_backward_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7n/c7nhsrbdbjanbvdd22sosthbe3sgaohc4nwwm3abippyodf5lo4u.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_1 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu, %primals_3], 1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 36)
    x1 = xindex // 36
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 36, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (4*x1 + ((-32) + x0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp4, tmp11, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ts/ctsytoacaibjwf652m5v5rhtfelf3gffp3kmrnvmzctmawqiiq2g.py
# Topologically Sorted Source Nodes: [linear_3, x_6], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear_3 => add_tensor_1
#   x_6 => relu_3
# Graph fragment:
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_9), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
triton_poi_fused_addmm_relu_2 = async_compile.triton('triton_poi_fused_addmm_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mq/cmqq23afkntyo3arb7m6wbvpz3cnxnxpymfbbascbe6maz4w4iii.py
# Topologically Sorted Source Nodes: [log_std, log_std_1], Original ATen: [aten.addmm, aten.clamp, aten.ge, aten.le, aten.logical_and]
# Source node to ATen node mapping:
#   log_std => add_tensor
#   log_std_1 => clamp_max, clamp_min
# Graph fragment:
#   %add_tensor : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_13), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_tensor, -20), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 2), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_tensor, -20), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_tensor, 2), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %le), kwargs = {})
triton_poi_fused_addmm_clamp_ge_le_logical_and_3 = async_compile.triton('triton_poi_fused_addmm_clamp_ge_le_logical_and_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clamp_ge_le_logical_and_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clamp_ge_le_logical_and_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -20.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 2.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 >= tmp3
    tmp8 = tmp2 <= tmp5
    tmp9 = tmp7 & tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
    args.clear()
    assert_size_stride(primals_1, (32, 4), (4, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (32, 36), (36, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, 36), (36, 1))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, 36), (36, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (4, 32), (32, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 32), (32, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (4, 32), (1, 4), 0), out=buf0)
        del primals_1
        buf14 = empty_strided_cuda((4, 32), (32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [linear, x], Original ATen: [aten.addmm, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_threshold_backward_0.run(buf0, primals_2, buf14, 128, grid=grid(128), stream=stream0)
        buf1 = empty_strided_cuda((4, 36), (36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf0, primals_2, primals_3, buf1, 144, grid=grid(144), stream=stream0)
        del primals_2
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf1, reinterpret_tensor(primals_4, (36, 32), (1, 36), 0), out=buf2)
        buf13 = empty_strided_cuda((4, 32), (32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [linear_1, x_2], Original ATen: [aten.addmm, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_threshold_backward_0.run(buf2, primals_5, buf13, 128, grid=grid(128), stream=stream0)
        buf3 = empty_strided_cuda((4, 36), (36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf2, primals_5, primals_3, buf3, 144, grid=grid(144), stream=stream0)
        del primals_5
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(buf3, reinterpret_tensor(primals_6, (36, 32), (1, 36), 0), out=buf4)
        buf12 = empty_strided_cuda((4, 32), (32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [linear_2, x_4], Original ATen: [aten.addmm, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_threshold_backward_0.run(buf4, primals_7, buf12, 128, grid=grid(128), stream=stream0)
        buf5 = empty_strided_cuda((4, 36), (36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf4, primals_7, primals_3, buf5, 144, grid=grid(144), stream=stream0)
        del primals_7
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.mm(buf5, reinterpret_tensor(primals_8, (36, 32), (1, 36), 0), out=buf6)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_3, x_6], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_2.run(buf7, primals_9, 128, grid=grid(128), stream=stream0)
        del primals_9
        buf8 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mu], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf7, reinterpret_tensor(primals_10, (32, 4), (1, 32), 0), alpha=1, beta=1, out=buf8)
        del primals_11
        buf9 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_std], Original ATen: [aten.addmm]
        extern_kernels.mm(buf7, reinterpret_tensor(primals_12, (32, 4), (1, 32), 0), out=buf9)
        buf10 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf11 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [log_std, log_std_1], Original ATen: [aten.addmm, aten.clamp, aten.ge, aten.le, aten.logical_and]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clamp_ge_le_logical_and_3.run(buf9, primals_13, buf10, buf11, 16, grid=grid(16), stream=stream0)
        del buf9
        del primals_13
    return (buf8, buf10, primals_3, buf1, buf3, buf5, buf7, buf11, primals_12, primals_10, primals_8, buf12, primals_6, buf13, primals_4, buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 36), (36, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 36), (36, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 36), (36, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
