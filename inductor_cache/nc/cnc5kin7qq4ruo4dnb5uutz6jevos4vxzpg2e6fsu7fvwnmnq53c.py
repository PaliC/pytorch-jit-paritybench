# AOT ID: ['19_forward']
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


# kernel path: inductor_cache/tk/ctkivoutnezfz4jmkssirb7mfrxxqnc5buivrolfgeog6fmfn6eq.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_1 => add, add_1, convert_element_type, convert_element_type_1, iota, mul, mul_1
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (124,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_0 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 124
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e6/ce6jw2y5hl3wgqoqupvnyjaf4tnatdsbxrkuza5yll35at6i76sq.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_2 => add_4, add_5, convert_element_type_4, convert_element_type_5, iota_2, mul_4, mul_5
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (248,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 0), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 0.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.5), kwargs = {})
#   %convert_element_type_5 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_1 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_1', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ad/cadshbg6noxmxhgdglrg6qgs6leecxam3u4ny5ynkg7ns7v56khw.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._unsafe_index]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => _unsafe_index
#   x_2 => _unsafe_index_1
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %unsqueeze, %convert_element_type_1]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, %unsqueeze_1, %convert_element_type_5]), kwargs = {})
triton_poi_fused__unsafe_index_convolution_2 = async_compile.triton('triton_poi_fused__unsafe_index_convolution_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_convolution_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_convolution_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3936256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 248) % 248)
    x0 = (xindex % 248)
    x5 = xindex // 61504
    x2 = ((xindex // 61504) % 16)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 124, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp4), None, eviction_policy='evict_last')
    tmp10 = tl.full([XBLOCK], 62, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tmp14 = tl.load(in_ptr1 + (tmp8), None, eviction_policy='evict_last')
    tmp15 = tmp14 + tmp10
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr2 + (tmp17 + 62*tmp13 + 3844*x5), None, eviction_policy='evict_last')
    tmp20 = tmp18 + tmp19
    tl.store(out_ptr0 + (x6), tmp20, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 10, 3, 3), (90, 9, 3, 1))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (4, 10, 64, 64), (40960, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 16, 62, 62), (61504, 3844, 62, 1))
        buf1 = empty_strided_cuda((124, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_0.run(buf1, 124, grid=grid(124), stream=stream0)
        buf2 = empty_strided_cuda((248, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_1.run(buf2, 248, grid=grid(248), stream=stream0)
        buf3 = empty_strided_cuda((4, 16, 248, 248), (984064, 61504, 248, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_convolution_2.run(buf2, buf1, buf0, primals_2, buf3, 3936256, grid=grid(3936256), stream=stream0)
        del buf0
        del primals_2
    return (buf3, primals_1, primals_3, buf1, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 10, 3, 3), (90, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 10, 64, 64), (40960, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
