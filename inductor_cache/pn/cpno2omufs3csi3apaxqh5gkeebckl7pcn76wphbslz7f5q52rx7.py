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


# kernel path: inductor_cache/5q/c5qg6oc54ns76rvzf5gxxyj6luemxzvlfwfiadylqueqhahwxhby.py
# Topologically Sorted Source Nodes: [mul, kernel_x, x, kernel_x_1, x_2, mul_2], Original ATen: [aten.mul, aten.full, aten.convolution]
# Source node to ATen node mapping:
#   kernel_x => full_default
#   kernel_x_1 => full_default_2
#   mul => mul
#   mul_2 => mul_2
#   x => convolution
#   x_2 => convolution_2
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 1, 1, 9], 0.1111111111111111), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %full_default, None, [1, 1], [0, 4], [1, 1], False, [0, 0], 4), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 1, 1, 9], 0.1111111111111111), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %full_default_2, None, [1, 1], [0, 4], [1, 1], False, [0, 0], 4), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg0_1), kwargs = {})
triton_poi_fused_convolution_full_mul_0 = async_compile.triton('triton_poi_fused_convolution_full_mul_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_full_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_full_mul_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x2 + 16*y3), xmask & ymask)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tl.store(out_ptr0 + (y0 + 4*x2 + 64*y1), tmp2, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 4*x2 + 64*y1), tmp0, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 4*x2 + 64*y1), tmp3, xmask & ymask)
    tl.store(out_ptr3 + (y0 + 4*x2 + 64*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/db/cdb3iir6oyoirtm4g4zhhd3a5ipp2ranm5fq7y6bzed5vbrlriw4.py
# Topologically Sorted Source Nodes: [kernel_x_2], Original ATen: [aten.full]
# Source node to ATen node mapping:
#   kernel_x_2 => full_default_4
# Graph fragment:
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 1, 1, 9], 0.1111111111111111), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_full_1 = async_compile.triton('triton_poi_fused_full_1', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_full_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_full_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.1111111111111111
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lz/clzdozvaaxqyyce4hs7j5mvezwwj2rb3hpgfhosc2i455pcgzoaa.py
# Topologically Sorted Source Nodes: [mul_1, cov_xy, mul_3, var_x, add, A, A_1, mul_5, mul_4, b, b_1, add_1], Original ATen: [aten.mul, aten.sub, aten.add, aten.div, aten._to_copy, aten.arange, aten.clamp, aten._unsafe_index]
# Source node to ATen node mapping:
#   A => div
#   A_1 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_3, add_5, add_6, add_7, clamp_max_2, clamp_max_3, clamp_min_1, clamp_min_2, clamp_min_3, convert_element_type_1, convert_element_type_2, convert_element_type_3, iota_1, mul_6, mul_7, mul_8, mul_9, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9
#   add => add
#   add_1 => add_15
#   b => sub_2
#   b_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_10, add_12, add_13, add_14, clamp_max_6, clamp_max_7, clamp_min_5, clamp_min_6, clamp_min_7, convert_element_type_5, convert_element_type_6, convert_element_type_7, iota_3, mul_11, mul_12, mul_13, mul_14, sub_11, sub_12, sub_13, sub_14, sub_15, sub_16
#   cov_xy => sub
#   mul_1 => mul_1
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_15
#   var_x => sub_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %mul_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, %convolution_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %mul_3), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_1, 1e-05), kwargs = {})
#   %div : [num_users=5] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %add), kwargs = {})
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 1.0), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_6, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_4, 0.0), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_1, torch.int64), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %clamp_max_2), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_8), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_2), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_7), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %add_5), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_8, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %clamp_max_3), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_9), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %arg2_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %convolution_1), kwargs = {})
#   %sub_2 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %mul_4), kwargs = {})
#   %convert_element_type_5 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
#   %iota_3 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_3, torch.float32), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 0.5), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 1.0), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_11, 0.5), kwargs = {})
#   %clamp_min_5 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_11, 0.0), kwargs = {})
#   %convert_element_type_7 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_5, torch.int64), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%sub_2, [None, None, %clamp_max_4, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%sub_2, [None, None, %clamp_max_4, %convert_element_type_7]), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_5, %convert_element_type_7), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_12, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %clamp_max_6), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_13), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%sub_2, [None, None, %convert_element_type_5, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%sub_2, [None, None, %convert_element_type_5, %convert_element_type_7]), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %clamp_max_6), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_12), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %add_12), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %convert_element_type_5), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_15, 0.0), kwargs = {})
#   %clamp_max_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 1.0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %clamp_max_7), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %mul_14), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %add_14), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_div_mul_sub_2 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_div_mul_sub_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_div_mul_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_div_mul_sub_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x5 = xindex
    tmp97 = tl.load(in_ptr4 + (x5), xmask)
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 + tmp2
    tmp17 = tmp16 * tmp4
    tmp18 = tmp17 - tmp2
    tmp19 = triton_helpers.maximum(tmp18, tmp7)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tmp20 + tmp10
    tmp22 = triton_helpers.minimum(tmp21, tmp12)
    tmp23 = tl.load(in_ptr0 + (x2 + 4*tmp22 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (x2 + 4*tmp22 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (x2 + 4*tmp22 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 - tmp26
    tmp28 = tl.load(in_ptr3 + (x2 + 4*tmp22 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp29 = tmp24 * tmp24
    tmp30 = tmp28 - tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = tmp27 / tmp32
    tmp34 = tl.load(in_ptr0 + (x2 + 4*tmp20 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (x2 + 4*tmp20 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (x2 + 4*tmp20 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp37 = tmp35 * tmp36
    tmp38 = tmp34 - tmp37
    tmp39 = tl.load(in_ptr3 + (x2 + 4*tmp20 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp40 = tmp35 * tmp35
    tmp41 = tmp39 - tmp40
    tmp42 = tmp41 + tmp31
    tmp43 = tmp38 / tmp42
    tmp44 = tl.load(in_ptr0 + (x2 + 4*tmp22 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr1 + (x2 + 4*tmp22 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr2 + (x2 + 4*tmp22 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp47 = tmp45 * tmp46
    tmp48 = tmp44 - tmp47
    tmp49 = tl.load(in_ptr3 + (x2 + 4*tmp22 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp50 = tmp45 * tmp45
    tmp51 = tmp49 - tmp50
    tmp52 = tmp51 + tmp31
    tmp53 = tmp48 / tmp52
    tmp54 = tl.load(in_ptr0 + (x2 + 4*tmp20 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr1 + (x2 + 4*tmp20 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr2 + (x2 + 4*tmp20 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp57 = tmp55 * tmp56
    tmp58 = tmp54 - tmp57
    tmp59 = tl.load(in_ptr3 + (x2 + 4*tmp20 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp60 = tmp55 * tmp55
    tmp61 = tmp59 - tmp60
    tmp62 = tmp61 + tmp31
    tmp63 = tmp58 / tmp62
    tmp64 = tmp53 - tmp63
    tmp65 = tmp20.to(tl.float32)
    tmp66 = tmp19 - tmp65
    tmp67 = triton_helpers.maximum(tmp66, tmp7)
    tmp68 = triton_helpers.minimum(tmp67, tmp4)
    tmp69 = tmp64 * tmp68
    tmp70 = tmp63 + tmp69
    tmp71 = tmp33 * tmp24
    tmp72 = tmp25 - tmp71
    tmp73 = tmp43 * tmp35
    tmp74 = tmp36 - tmp73
    tmp75 = tmp53 * tmp45
    tmp76 = tmp46 - tmp75
    tmp77 = tmp63 * tmp55
    tmp78 = tmp56 - tmp77
    tmp79 = tmp76 - tmp78
    tmp80 = tmp79 * tmp68
    tmp81 = tmp78 + tmp80
    tmp82 = tmp33 - tmp43
    tmp83 = tmp82 * tmp68
    tmp84 = tmp43 + tmp83
    tmp85 = tmp84 - tmp70
    tmp86 = tmp9.to(tl.float32)
    tmp87 = tmp8 - tmp86
    tmp88 = triton_helpers.maximum(tmp87, tmp7)
    tmp89 = triton_helpers.minimum(tmp88, tmp4)
    tmp90 = tmp85 * tmp89
    tmp91 = tmp72 - tmp74
    tmp92 = tmp91 * tmp68
    tmp93 = tmp74 + tmp92
    tmp94 = tmp93 - tmp81
    tmp95 = tmp94 * tmp89
    tmp96 = tmp70 + tmp90
    tmp98 = tmp96 * tmp97
    tmp99 = tmp81 + tmp95
    tmp100 = tmp98 + tmp99
    tl.store(in_out_ptr0 + (x5), tmp100, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf6 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf15 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf11 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mul, kernel_x, x, kernel_x_1, x_2, mul_2], Original ATen: [aten.mul, aten.full, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_full_mul_0.run(arg0_1, arg1_1, buf0, buf6, buf15, buf11, 16, 16, grid=grid(16, 16), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((4, 1, 1, 9), (9, 9, 9, 1), torch.float32)
        # Topologically Sorted Source Nodes: [kernel_x_2], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_1.run(buf1, 36, grid=grid(36), stream=stream0)
        # Topologically Sorted Source Nodes: [mul, kernel_x_2, x_4], Original ATen: [aten.mul, aten.full, aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(0, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf2, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf0
        buf3 = reinterpret_tensor(buf1, (4, 1, 9, 1), (9, 9, 1, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [kernel_y_2], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_1.run(buf3, 36, grid=grid(36), stream=stream0)
        # Topologically Sorted Source Nodes: [kernel_y_2, x_5], Original ATen: [aten.full, aten.convolution]
        buf4 = extern_kernels.convolution(buf2, buf3, stride=(1, 1), padding=(4, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf4, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf2
        buf5 = reinterpret_tensor(buf3, (4, 1, 1, 9), (9, 9, 9, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [kernel_x], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_1.run(buf5, 36, grid=grid(36), stream=stream0)
        # Topologically Sorted Source Nodes: [kernel_x, x], Original ATen: [aten.full, aten.convolution]
        buf7 = extern_kernels.convolution(buf6, buf5, stride=(1, 1), padding=(0, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf7, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf6
        buf8 = reinterpret_tensor(buf5, (4, 1, 9, 1), (9, 9, 1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [kernel_y], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_1.run(buf8, 36, grid=grid(36), stream=stream0)
        # Topologically Sorted Source Nodes: [kernel_y, x_1], Original ATen: [aten.full, aten.convolution]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(4, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf9, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf7
        buf10 = reinterpret_tensor(buf8, (4, 1, 1, 9), (9, 9, 9, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [kernel_x_1], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_1.run(buf10, 36, grid=grid(36), stream=stream0)
        # Topologically Sorted Source Nodes: [kernel_x_1, x_2], Original ATen: [aten.full, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, buf10, stride=(1, 1), padding=(0, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf12, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf11
        buf13 = reinterpret_tensor(buf10, (4, 1, 9, 1), (9, 9, 1, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [kernel_y_1], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_1.run(buf13, 36, grid=grid(36), stream=stream0)
        # Topologically Sorted Source Nodes: [kernel_y_1, x_3], Original ATen: [aten.full, aten.convolution]
        buf14 = extern_kernels.convolution(buf12, buf13, stride=(1, 1), padding=(4, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf14, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf12
        buf16 = reinterpret_tensor(buf13, (4, 1, 1, 9), (9, 9, 9, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [kernel_x_3], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_1.run(buf16, 36, grid=grid(36), stream=stream0)
        # Topologically Sorted Source Nodes: [mul_2, kernel_x_3, x_6], Original ATen: [aten.mul, aten.full, aten.convolution]
        buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(0, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf17, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf15
        buf18 = reinterpret_tensor(buf16, (4, 1, 9, 1), (9, 9, 1, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [kernel_y_3], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_1.run(buf18, 36, grid=grid(36), stream=stream0)
        # Topologically Sorted Source Nodes: [kernel_y_3, x_7], Original ATen: [aten.full, aten.convolution]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(4, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf19, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf18
        buf22 = reinterpret_tensor(buf17, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf17  # reuse
        buf23 = buf22; del buf22  # reuse
        buf24 = buf23; del buf23  # reuse
        buf32 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [mul_1, cov_xy, mul_3, var_x, add, A, A_1, mul_5, mul_4, b, b_1, add_1], Original ATen: [aten.mul, aten.sub, aten.add, aten.div, aten._to_copy, aten.arange, aten.clamp, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_div_mul_sub_2.run(buf32, buf4, buf9, buf14, buf19, arg2_1, 256, grid=grid(256), stream=stream0)
        del arg2_1
        del buf14
        del buf19
        del buf4
        del buf9
    return (buf32, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
