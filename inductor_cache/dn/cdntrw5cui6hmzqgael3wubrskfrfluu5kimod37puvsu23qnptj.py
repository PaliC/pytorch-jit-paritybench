# AOT ID: ['20_inference']
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


# kernel path: inductor_cache/tz/ctzyz4pnb6aikt2rlf5apgchevowlwb7rugf5fqyletij37nv63q.py
# Topologically Sorted Source Nodes: [known_mask, pad], Original ATen: [aten.rsub, aten.replication_pad2d]
# Source node to ATen node mapping:
#   known_mask => sub
#   pad => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg0_1), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%sub, [None, None, %clamp_max, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %clamp_max_1]), kwargs = {})
triton_poi_fused_replication_pad2d_rsub_0 = async_compile.triton('triton_poi_fused_replication_pad2d_rsub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_replication_pad2d_rsub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_replication_pad2d_rsub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 8)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4*((3) * ((3) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < (3))) + 16*x2 + ((3) * ((3) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (3)))), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sf/csfkqrqonzgtc5uvjgtaao7lm5i3yrsdna22yd57efeom4tplthp.py
# Topologically Sorted Source Nodes: [gt, dilated_known_mask, sub_1, pad_1], Original ATen: [aten.gt, aten._to_copy, aten.rsub, aten.replication_pad2d]
# Source node to ATen node mapping:
#   dilated_known_mask => convert_element_type
#   gt => gt
#   pad_1 => _unsafe_index_2, _unsafe_index_3
#   sub_1 => sub_1
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 1), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%sub_1, [None, None, %clamp_max_2, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %clamp_max_3]), kwargs = {})
triton_poi_fused__to_copy_gt_replication_pad2d_rsub_1 = async_compile.triton('triton_poi_fused__to_copy_gt_replication_pad2d_rsub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_gt_replication_pad2d_rsub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_gt_replication_pad2d_rsub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 8)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4*((3) * ((3) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < (3))) + 16*x2 + ((3) * ((3) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (3)))), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f3/cf3u6eivz2cb4bkcq6s6dg5fflvzgqogs7hjgs2nke7r3jgczoay.py
# Topologically Sorted Source Nodes: [result], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   result => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, %arg0_1), kwargs = {})
triton_poi_fused_mul_2 = async_compile.triton('triton_poi_fused_mul_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 1, 4, 4), (16, 16, 4, 1))
    assert_size_stride(arg1_1, (1, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg2_1, (1, 1, 5, 5), (25, 25, 5, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 8, 8), (64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [known_mask, pad], Original ATen: [aten.rsub, aten.replication_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_replication_pad2d_rsub_0.run(arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [known_mask, pad, conv2d], Original ATen: [aten.rsub, aten.replication_pad2d, aten.convolution]
        buf1 = extern_kernels.convolution(buf0, arg1_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 1, 4, 4), (16, 16, 4, 1))
        del arg1_1
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [gt, dilated_known_mask, sub_1, pad_1], Original ATen: [aten.gt, aten._to_copy, aten.rsub, aten.replication_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_gt_replication_pad2d_rsub_1.run(buf1, buf2, 256, grid=grid(256), stream=stream0)
        del buf1
        # Topologically Sorted Source Nodes: [gt, dilated_known_mask, sub_1, pad_1, conv2d_1], Original ATen: [aten.gt, aten._to_copy, aten.rsub, aten.replication_pad2d, aten.convolution]
        buf3 = extern_kernels.convolution(buf2, arg2_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 1, 4, 4), (16, 16, 4, 1))
        del arg2_1
        del buf2
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [result], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_2.run(buf4, arg0_1, 64, grid=grid(64), stream=stream0)
        del arg0_1
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 1, 4, 4), (16, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
