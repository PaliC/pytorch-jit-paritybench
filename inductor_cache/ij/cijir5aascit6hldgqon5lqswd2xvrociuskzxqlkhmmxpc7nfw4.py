# AOT ID: ['29_inference']
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


# kernel path: inductor_cache/fh/cfhywsehnpgcklbbtzdsctdufnhx5f6pjfqygtklojidgfrntdml.py
# Topologically Sorted Source Nodes: [dim_t, floordiv, mul_2, truediv_2, dim_t_1, pos_y], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.div, aten.pow]
# Source node to ATen node mapping:
#   dim_t => add_2, convert_element_type, iota, mul_2
#   dim_t_1 => pow_1
#   floordiv => div_2
#   mul_2 => mul_3
#   pos_y => div_5
#   truediv_2 => div_3
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%convert_element_type, 2), kwargs = {rounding_mode: floor})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 2), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, 128), kwargs = {})
#   %pow_1 : [num_users=2] = call_function[target=torch.ops.aten.pow.Scalar](args = (10000, %div_3), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%unsqueeze_1, %pow_1), kwargs = {})
triton_poi_fused_arange_div_floor_divide_mul_pow_0 = async_compile.triton('triton_poi_fused_arange_div_floor_divide_mul_pow_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_div_floor_divide_mul_pow_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_div_floor_divide_mul_pow_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 512) % 4)
    x0 = (xindex % 128)
    x5 = xindex
    tmp0 = 1 + x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 4.000001
    tmp5 = tmp3 / tmp4
    tmp6 = 6.283185307179586
    tmp7 = tmp5 * tmp6
    tmp8 = x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 0.5
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.floor(tmp11)
    tmp13 = 2.0
    tmp14 = tmp12 * tmp13
    tmp15 = 0.0078125
    tmp16 = tmp14 * tmp15
    tmp17 = 10000.0
    tmp18 = libdevice.pow(tmp17, tmp16)
    tmp19 = tmp7 / tmp18
    tl.store(out_ptr0 + (x5), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/kp/ckp5xow7uh3wlgmqu3enicq4ilh765ca2ecepsedih4q2hzou6pf.py
# Topologically Sorted Source Nodes: [dim_t, floordiv, mul_2, truediv_2, dim_t_1, pos_x], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.div, aten.pow]
# Source node to ATen node mapping:
#   dim_t => add_2, convert_element_type, iota, mul_2
#   dim_t_1 => pow_1
#   floordiv => div_2
#   mul_2 => mul_3
#   pos_x => div_4
#   truediv_2 => div_3
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%convert_element_type, 2), kwargs = {rounding_mode: floor})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 2), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, 128), kwargs = {})
#   %pow_1 : [num_users=2] = call_function[target=torch.ops.aten.pow.Scalar](args = (10000, %div_3), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%unsqueeze, %pow_1), kwargs = {})
triton_poi_fused_arange_div_floor_divide_mul_pow_1 = async_compile.triton('triton_poi_fused_arange_div_floor_divide_mul_pow_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_div_floor_divide_mul_pow_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_div_floor_divide_mul_pow_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 128) % 4)
    x0 = (xindex % 128)
    x3 = xindex
    tmp0 = 1 + x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 4.000001
    tmp5 = tmp3 / tmp4
    tmp6 = 6.283185307179586
    tmp7 = tmp5 * tmp6
    tmp8 = x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 0.5
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.floor(tmp11)
    tmp13 = 2.0
    tmp14 = tmp12 * tmp13
    tmp15 = 0.0078125
    tmp16 = tmp14 * tmp15
    tmp17 = 10000.0
    tmp18 = libdevice.pow(tmp17, tmp16)
    tmp19 = tmp7 / tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/zz/czzdvfukju55ss2xjqq3pkvb2w3ugcsmbqgelsfmol64c26umzuu.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_1, %view], 3), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = ((x0) % 2)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (2*((((x0) // 2) % 64)) + 128*x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl_math.sin(tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = tmp5 >= tmp8
    tmp16 = tl.full([1], 2, tl.int64)
    tmp17 = tmp5 < tmp16
    tmp18 = tmp15 & tmp4
    tmp19 = tl.load(in_ptr0 + (1 + 2*((((x0) // 2) % 64)) + 128*x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl_math.cos(tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp9, tmp14, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp4, tmp23, tmp24)
    tmp26 = tmp0 >= tmp3
    tmp27 = tl.full([1], 256, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = (((-128) + x0) % 2)
    tmp30 = tl.full([1], 0, tl.int64)
    tmp31 = tmp29 >= tmp30
    tmp32 = tl.full([1], 1, tl.int64)
    tmp33 = tmp29 < tmp32
    tmp34 = tmp33 & tmp26
    tmp35 = tl.load(in_ptr1 + (2*(((((-128) + x0) // 2) % 64)) + 128*x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp36 = tl_math.sin(tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp34, tmp36, tmp37)
    tmp39 = tmp29 >= tmp32
    tmp40 = tl.full([1], 2, tl.int64)
    tmp41 = tmp29 < tmp40
    tmp42 = tmp39 & tmp26
    tmp43 = tl.load(in_ptr1 + (1 + 2*(((((-128) + x0) // 2) % 64)) + 128*x1), tmp42, eviction_policy='evict_last', other=0.0)
    tmp44 = tl_math.cos(tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp42, tmp44, tmp45)
    tmp47 = tl.where(tmp33, tmp38, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp26, tmp47, tmp48)
    tmp50 = tl.where(tmp4, tmp25, tmp49)
    tl.store(out_ptr0 + (x2), tmp50, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 128), (2048, 512, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dim_t, floordiv, mul_2, truediv_2, dim_t_1, pos_y], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_div_floor_divide_mul_pow_0.run(buf0, 8192, grid=grid(8192), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 4, 128), (2048, 512, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dim_t, floordiv, mul_2, truediv_2, dim_t_1, pos_x], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_div_floor_divide_mul_pow_1.run(buf1, 8192, grid=grid(8192), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 4, 256), (4096, 1024, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf0, buf1, buf2, 16384, grid=grid(16384), stream=stream0)
        del buf0
        del buf1
    return (reinterpret_tensor(buf2, (4, 256, 4, 4), (4096, 1, 1024, 256), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    fn = lambda: call([])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
