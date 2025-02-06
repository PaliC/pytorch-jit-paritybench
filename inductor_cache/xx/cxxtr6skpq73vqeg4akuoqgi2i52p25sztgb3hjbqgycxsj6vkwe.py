# AOT ID: ['3_inference']
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


# kernel path: inductor_cache/zc/czcpe6eonivhfhna6mqugsqge2zl43o5j2u2k6ab6osbinjl23b7.py
# Topologically Sorted Source Nodes: [x_grid, copy_, copy__1, fill_], Original ATen: [aten.linspace, aten.copy, aten.fill]
# Source node to ATen node mapping:
#   copy_ => copy
#   copy__1 => copy_1
#   fill_ => full_default
#   x_grid => add, convert_element_type, convert_element_type_1, iota, lt, mul, mul_1, sub, sub_1, where
# Graph fragment:
#   %iota : [num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota, 32.0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 1.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, -31.5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (63, %iota), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, 1.0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (31.5, %mul_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt, %add, %sub_1), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %where), kwargs = {})
#   %select_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%empty, %copy, 3, 0), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_3, %unsqueeze), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %copy_1, 3, 1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 64], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %full_default, 3, 2), kwargs = {})
triton_poi_fused_copy_fill_linspace_0 = async_compile.triton('triton_poi_fused_copy_fill_linspace_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_fill_linspace_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_fill_linspace_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 3)
    x2 = xindex // 192
    x1 = ((xindex // 3) % 64)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 32.0
    tmp8 = tmp6 < tmp7
    tmp9 = 1.0
    tmp10 = tmp6 * tmp9
    tmp11 = -31.5
    tmp12 = tmp10 + tmp11
    tmp13 = 63 + ((-1)*x2)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp9
    tmp16 = 31.5
    tmp17 = tmp16 - tmp15
    tmp18 = tl.where(tmp8, tmp12, tmp17)
    tmp19 = tl.full([1], 0, tl.int32)
    tmp20 = tmp0 == tmp19
    tmp21 = x1
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 < tmp7
    tmp24 = tmp22 * tmp9
    tmp25 = tmp24 + tmp11
    tmp26 = 63 + ((-1)*x1)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp9
    tmp29 = tmp16 - tmp28
    tmp30 = tl.where(tmp23, tmp25, tmp29)
    tmp31 = float("nan")
    tmp32 = tl.where(tmp20, tmp30, tmp31)
    tmp33 = tl.where(tmp4, tmp18, tmp32)
    tmp34 = tl.where(tmp2, tmp9, tmp33)
    tl.store(out_ptr0 + (x4), tmp34, None)
''', device_str='cuda')


# kernel path: inductor_cache/3k/c3kw5fpxgzjczfzjk5dlwbabpanuivsyttluk3twbgqszb6krszn.py
# Topologically Sorted Source Nodes: [tensor_1, rescaled_theta], Original ATen: [aten.lift_fresh, aten.div]
# Source node to ATen node mapping:
#   rescaled_theta => div
#   tensor_1 => lift_fresh_copy_1
# Graph fragment:
#   %lift_fresh_copy_1 : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%permute, %lift_fresh_copy_1), kwargs = {})
triton_poi_fused_div_lift_fresh_1 = async_compile.triton('triton_poi_fused_div_lift_fresh_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_lift_fresh_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_lift_fresh_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = x1 + 3*x0
    tmp1 = tl.full([1], 3, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 2, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = 0.35973861813545227
    tmp8 = 0.0
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = 0.9330531358718872
    tmp11 = tl.where(tmp4, tmp10, tmp9)
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.full([1], 5, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.where(tmp15, tmp10, tmp8)
    tmp17 = -0.35973861813545227
    tmp18 = tl.where(tmp13, tmp17, tmp16)
    tmp19 = tl.where(tmp2, tmp11, tmp18)
    tmp20 = x0
    tmp21 = tmp20 < tmp3
    tmp22 = 32.0
    tmp23 = tl.where(tmp21, tmp22, tmp22)
    tmp24 = tmp19 / tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 64, 64, 3), (12288, 192, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_grid, copy_, copy__1, fill_], Original ATen: [aten.linspace, aten.copy, aten.fill]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_fill_linspace_0.run(buf1, 12288, grid=grid(12288), stream=stream0)
        buf2 = empty_strided_cuda((1, 3, 2), (6, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensor_1, rescaled_theta], Original ATen: [aten.lift_fresh, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_lift_fresh_1.run(buf2, 6, grid=grid(6), stream=stream0)
        buf3 = empty_strided_cuda((1, 4096, 2), (8192, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensor_1, rescaled_theta, output_grid], Original ATen: [aten.lift_fresh, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1, (1, 4096, 3), (0, 3, 1), 0), buf2, out=buf3)
        del buf1
        del buf2
    return (reinterpret_tensor(buf3, (1, 64, 64, 2), (8192, 128, 2, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    fn = lambda: call([])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
