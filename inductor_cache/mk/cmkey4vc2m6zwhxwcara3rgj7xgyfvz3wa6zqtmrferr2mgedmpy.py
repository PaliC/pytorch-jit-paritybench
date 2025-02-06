# AOT ID: ['44_forward']
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


# kernel path: inductor_cache/yr/cyri5puopb75qgsijlw6g7o2ivyk66amgsv7mjoi5oswinzn3ooj.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input], Original ATen: [aten.sub, aten.div]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
triton_poi_fused_div_sub_0 = async_compile.triton('triton_poi_fused_div_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 262144}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 262144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 262144*y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + 262144*y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tmp6 = tmp5 - tmp1
    tmp7 = tmp6 / tmp3
    tl.store(out_ptr0 + (y0 + 3*x2 + 786432*y1), tmp4, ymask)
    tl.store(out_ptr1 + (y0 + 3*x2 + 786432*y1), tmp7, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/2o/c2o3hdg4jltwjfoqlaqywbuba2lgg43wdnnynpe2fupgou4xktfc.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_31], Original ATen: [aten.sub, aten.div, aten.convolution]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_31 => convolution_13
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_sub_1 = async_compile.triton('triton_poi_fused_convolution_div_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_sub_1(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ne/cne3yaq37y2refg7u2zut6adtgnnoasidtxn6uwwf66jt4ukzasa.py
# Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   in0_input => div
#   input_1 => convolution
#   input_2 => relu
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_div_relu_sub_2 = async_compile.triton('triton_poi_fused_convolution_div_relu_sub_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_relu_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_relu_sub_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/xx/cxxstfzajvwix64bp3j2covxeijgoqeee4jy6fp5gebmkvxaig6f.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_31, input_32, input_33], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_relu_sub_3 = async_compile.triton('triton_poi_fused_convolution_div_relu_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_relu_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_relu_sub_3(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cikukn7ysiu5mudtsv4guw4abtyturbcwx7uyk32idcq24k4r2ok.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_31, input_32, input_33, input_34, pow_1, sum_1, norm_factor, add, truediv_2, pow_2, sum_2, norm_factor_1, add_1, truediv_3, sub_2, pow_3], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.pow, aten.sum, aten.sqrt, aten.add]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_4 => relu_1
#   norm_factor => sqrt
#   norm_factor_1 => sqrt_1
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sum_1 => sum_1
#   sum_2 => sum_2
#   truediv_2 => div_2
#   truediv_3 => div_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt, 1e-10), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_1, %add), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_14, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [1], True), kwargs = {})
#   %sqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_2,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_1, 1e-10), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_14, %add_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_2, %div_3), kwargs = {})
#   %pow_3 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
triton_per_fused_add_convolution_div_pow_relu_sqrt_sub_sum_4 = async_compile.triton('triton_per_fused_add_convolution_div_pow_relu_sqrt_sub_sum_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_pow_relu_sqrt_sub_sum_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_pow_relu_sqrt_sub_sum_4(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 64*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr1 + (r1 + 64*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp5 + tmp1
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tmp8 = tmp4 * tmp4
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tmp7 * tmp7
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp16 = libdevice.sqrt(tmp11)
    tmp17 = 1e-10
    tmp18 = tmp16 + tmp17
    tmp19 = tmp4 / tmp18
    tmp20 = libdevice.sqrt(tmp15)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp7 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tl.store(in_out_ptr0 + (r1 + 64*x0), tmp4, None)
    tl.store(in_out_ptr1 + (r1 + 64*x0), tmp7, None)
    tl.store(out_ptr2 + (r1 + 64*x0), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/o5/co5hbjjdezz2mvro4xoietdq5zoiqzcu74soz64x6un2lp5z54jp.py
# Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_5 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 256)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 65536*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 65536*x2), None)
    tmp3 = tl.load(in_ptr0 + (32768 + x0 + 128*x1 + 65536*x2), None)
    tmp5 = tl.load(in_ptr0 + (32832 + x0 + 128*x1 + 65536*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/yq/cyqbfhiqvxxni26xrpnlr45a4clibhpefzrjkva3stib4jll3laz.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_31, input_32, input_33, input_34, input_35, input_36], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_6 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_6(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ia/ciappt7gnpunjwhdbwainhxofndgmaa2xotq4qyhn65eodizvxkc.py
# Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_7 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/ax/caxyqvnfriuh4ogv6lt52gewlccoi5cwfwd77y5543x2ddparfdq.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zi/czis5by4mjhtu7sxzklep434catysk7efeyqdayxi7iveiqovenz.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, pow_4, sum_3, norm_factor_2, add_2, truediv_4, pow_5, sum_4, norm_factor_3, add_3, truediv_5, sub_3, pow_6], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   norm_factor_2 => sqrt_2
#   norm_factor_3 => sqrt_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_6 => pow_6
#   sub => sub
#   sub_1 => sub_1
#   sub_3 => sub_3
#   sum_3 => sum_3
#   sum_4 => sum_4
#   truediv_4 => div_4
#   truediv_5 => div_5
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_3, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_4, [1], True), kwargs = {})
#   %sqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_3,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_2, 1e-10), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_3, %add_2), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_16, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1], True), kwargs = {})
#   %sqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_4,), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_3, 1e-10), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_16, %add_3), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_4, %div_5), kwargs = {})
#   %pow_6 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_3, 2), kwargs = {})
triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_9 = async_compile.triton('triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_9(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 128*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr1 + (r1 + 128*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp5 + tmp1
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tmp8 = tmp4 * tmp4
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tmp7 * tmp7
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp16 = libdevice.sqrt(tmp11)
    tmp17 = 1e-10
    tmp18 = tmp16 + tmp17
    tmp19 = tmp4 / tmp18
    tmp20 = libdevice.sqrt(tmp15)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp7 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp4, None)
    tl.store(in_out_ptr1 + (r1 + 128*x0), tmp7, None)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/2s/c2styv2wgkh2hkgul3fnkuniy6rlj3vp4g3hgsv63krzgkg6wd3b.py
# Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 128)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 65536*x2), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 65536*x2), None)
    tmp3 = tl.load(in_ptr0 + (32768 + x0 + 256*x1 + 65536*x2), None)
    tmp5 = tl.load(in_ptr0 + (32896 + x0 + 256*x1 + 65536*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/j6/cj6okdpio3yriecrga3w4emaw6bte2ptzjiuamgbx7anopjftrxi.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_2 => relu
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_11 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_11(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ym/cymfdb4iarcrolubehefivxafx5z7kkh5ifufjlkca5wj474de25.py
# Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_12 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/2g/c2g4jawo62jlubkgklccq6vzsou5kcpdvrpvta46vtvgjcqk3ktp.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_2 => relu
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_13 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_13(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/4r/c4rqwc7m6ly2wjiyszhahelgo7qhu3kzwowyqethpo3eb34jqofk.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, pow_7, sum_5, norm_factor_4, add_4, truediv_6, pow_8, sum_6, norm_factor_5, add_5, truediv_7, sub_4, pow_9], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
# Source node to ATen node mapping:
#   add_4 => add_4
#   add_5 => add_5
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_2 => relu
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   norm_factor_4 => sqrt_4
#   norm_factor_5 => sqrt_5
#   pow_7 => pow_7
#   pow_8 => pow_8
#   pow_9 => pow_9
#   sub => sub
#   sub_1 => sub_1
#   sub_4 => sub_4
#   sum_5 => sum_5
#   sum_6 => sum_6
#   truediv_6 => div_6
#   truediv_7 => div_7
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_6, 2), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1], True), kwargs = {})
#   %sqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_5,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_4, 1e-10), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_6, %add_4), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_19, 2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_8, [1], True), kwargs = {})
#   %sqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_6,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_5, 1e-10), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_19, %add_5), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_6, %div_7), kwargs = {})
#   %pow_9 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_14 = async_compile.triton('triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_14(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 65536
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 256*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr1 + (r1 + 256*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp5 + tmp1
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tmp8 = tmp4 * tmp4
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp7 * tmp7
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = libdevice.sqrt(tmp11)
    tmp17 = 1e-10
    tmp18 = tmp16 + tmp17
    tmp19 = tmp4 / tmp18
    tmp20 = libdevice.sqrt(tmp15)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp7 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tl.store(in_out_ptr0 + (r1 + 256*x0), tmp4, None)
    tl.store(in_out_ptr1 + (r1 + 256*x0), tmp7, None)
    tl.store(out_ptr2 + (r1 + 256*x0), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/pd/cpdlpxfecasf5o74t6ogwdovxgaha6ltpwkuoh62uk6k3hg44tdy.py
# Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_15 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 64)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 65536*x2), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 65536*x2), None)
    tmp3 = tl.load(in_ptr0 + (32768 + x0 + 512*x1 + 65536*x2), None)
    tmp5 = tl.load(in_ptr0 + (33024 + x0 + 512*x1 + 65536*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/lf/clfvf346bu65lmgnlugdnhixocl7hvlvkqm64a6n5cyc7nucoin2.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_2 => relu
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_47 => _low_memory_max_pool2d_with_offsets_6
#   input_48 => convolution_20
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ez/ceza3y5xfc7jndl7pdvzbyszp6h2bqp34mcxjtpfk7yilpzporgs.py
# Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/er/cer4ylyllexdylcodp4ogmjqxiqcagcqkklb6q5mu3kecdcjo5vd.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_47 => _low_memory_max_pool2d_with_offsets_6
#   input_48 => convolution_20
#   input_49 => relu_20
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_50 => convolution_21
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ix/cixfuffzp4msbatopv7nfzbo4vkomrwkhgl4coc3shliq5j2l56y.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, pow_10, sum_7, norm_factor_6, add_6, truediv_8, pow_11, sum_8, norm_factor_7, add_7, truediv_9, sub_5, pow_12], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
# Source node to ATen node mapping:
#   add_6 => add_6
#   add_7 => add_7
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_47 => _low_memory_max_pool2d_with_offsets_6
#   input_48 => convolution_20
#   input_49 => relu_20
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_50 => convolution_21
#   input_51 => relu_21
#   input_52 => convolution_22
#   input_53 => relu_22
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   norm_factor_6 => sqrt_6
#   norm_factor_7 => sqrt_7
#   pow_10 => pow_10
#   pow_11 => pow_11
#   pow_12 => pow_12
#   sub => sub
#   sub_1 => sub_1
#   sub_5 => sub_5
#   sum_7 => sum_7
#   sum_8 => sum_8
#   truediv_8 => div_8
#   truediv_9 => div_9
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_9, 2), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_10, [1], True), kwargs = {})
#   %sqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_7,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_6, 1e-10), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_9, %add_6), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_22, 2), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1], True), kwargs = {})
#   %sqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_8,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_7, 1e-10), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_22, %add_7), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_8, %div_9), kwargs = {})
#   %pow_12 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_5, 2), kwargs = {})
triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_19 = async_compile.triton('triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_19(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr1 + (r1 + 512*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp5 + tmp1
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tmp8 = tmp4 * tmp4
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp7 * tmp7
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = libdevice.sqrt(tmp11)
    tmp17 = 1e-10
    tmp18 = tmp16 + tmp17
    tmp19 = tmp4 / tmp18
    tmp20 = libdevice.sqrt(tmp15)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp7 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tl.store(in_out_ptr0 + (r1 + 512*x0), tmp4, None)
    tl.store(in_out_ptr1 + (r1 + 512*x0), tmp7, None)
    tl.store(out_ptr2 + (r1 + 512*x0), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/x2/cx2sfy5hhmcsndppu3l6gapmbmvh7x2lhnzekxwzpgr3vw3ziqln.py
# Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => _low_memory_max_pool2d_with_offsets_3
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 32)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1 + 65536*x2), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + 1024*x1 + 65536*x2), None)
    tmp3 = tl.load(in_ptr0 + (32768 + x0 + 1024*x1 + 65536*x2), None)
    tmp5 = tl.load(in_ptr0 + (33280 + x0 + 1024*x1 + 65536*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/jz/cjzsio35x5vil76jrncytrikvcw7whpfuib7bulbenyiudtzbp67.py
# Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in0_input => div
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => _low_memory_max_pool2d_with_offsets_3
#   input_25 => convolution_10
#   input_26 => relu_10
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %primals_25, %primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_21 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/pv/cpvao3y4bwhga2eot2agjxkdl4zy5xt2pmxluo76zlnpmabtknv2.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, pow_13, sum_9, norm_factor_8, add_8, truediv_10, pow_14, sum_10, norm_factor_9, add_9, truediv_11, sub_6, pow_15], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
# Source node to ATen node mapping:
#   add_8 => add_8
#   add_9 => add_9
#   in0_input => div
#   in1_input => div_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => _low_memory_max_pool2d_with_offsets_3
#   input_25 => convolution_10
#   input_26 => relu_10
#   input_27 => convolution_11
#   input_28 => relu_11
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_47 => _low_memory_max_pool2d_with_offsets_6
#   input_48 => convolution_20
#   input_49 => relu_20
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_50 => convolution_21
#   input_51 => relu_21
#   input_52 => convolution_22
#   input_53 => relu_22
#   input_54 => _low_memory_max_pool2d_with_offsets_7
#   input_55 => convolution_23
#   input_56 => relu_23
#   input_57 => convolution_24
#   input_58 => relu_24
#   input_59 => convolution_25
#   input_6 => convolution_2
#   input_60 => relu_25
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   norm_factor_8 => sqrt_8
#   norm_factor_9 => sqrt_9
#   pow_13 => pow_13
#   pow_14 => pow_14
#   pow_15 => pow_15
#   sub => sub
#   sub_1 => sub_1
#   sub_6 => sub_6
#   sum_10 => sum_10
#   sum_9 => sum_9
#   truediv_10 => div_10
#   truediv_11 => div_11
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %primals_25, %primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %primals_27, %primals_28, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_29, %primals_30, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_7 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_14, %primals_25, %primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %primals_27, %primals_28, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_24 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %primals_29, %primals_30, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_12, 2), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_13, [1], True), kwargs = {})
#   %sqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_9,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_8, 1e-10), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_12, %add_8), kwargs = {})
#   %pow_14 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_25, 2), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_14, [1], True), kwargs = {})
#   %sqrt_9 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_10,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_9, 1e-10), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_25, %add_9), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_10, %div_11), kwargs = {})
#   %pow_15 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_6, 2), kwargs = {})
triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_22 = async_compile.triton('triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_22(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (r1 + 512*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp9 + tmp1
    tmp11 = triton_helpers.maximum(tmp3, tmp10)
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = libdevice.sqrt(tmp8)
    tmp17 = 1e-10
    tmp18 = tmp16 + tmp17
    tmp19 = tmp4 / tmp18
    tmp20 = libdevice.sqrt(tmp15)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp11 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tl.store(in_out_ptr0 + (r1 + 512*x0), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/7j/c7j77ldiyqlkw6twrylhs6fknivk6j3f7ht477tzx2z4y7b2dzdx.py
# Topologically Sorted Source Nodes: [val], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   val => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_26, [2, 3], True), kwargs = {})
triton_red_fused_mean_23 = async_compile.triton('triton_red_fused_mean_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_23(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rs/crs5zkhgu5wuzxjwfo7aqwcp6mzcvkis7ugzxxivn7cjixbbgnfa.py
# Topologically Sorted Source Nodes: [val], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   val => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_26, [2, 3], True), kwargs = {})
triton_per_fused_mean_24 = async_compile.triton('triton_per_fused_mean_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_24(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jt/cjtqraoaoqkpesxrgcp3zod5sxmclr5djnatg3ljvuf64aah3sdo.py
# Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_1 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_27, [2, 3], True), kwargs = {})
triton_red_fused_mean_25 = async_compile.triton('triton_red_fused_mean_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_25(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6l/c6lpk3uyhjxfwqhql3hbaqpsuyegjqcgwtnprqldp64u23bqhtlf.py
# Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_1 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_27, [2, 3], True), kwargs = {})
triton_per_fused_mean_26 = async_compile.triton('triton_per_fused_mean_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_26(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 8*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jt/cjtqwzjk2qxzynih66npkvlmk5sbs3yetjpp3v2aqcjj6mk3u6jz.py
# Topologically Sorted Source Nodes: [mean_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
# Graph fragment:
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_28, [2, 3], True), kwargs = {})
triton_red_fused_mean_27 = async_compile.triton('triton_red_fused_mean_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_27(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qw/cqw7bomahaukfaxzdd4kxciwkktoqhjbkwm7cu6dhq67mh4v6esv.py
# Topologically Sorted Source Nodes: [mean_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
# Graph fragment:
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_28, [2, 3], True), kwargs = {})
triton_per_fused_mean_28 = async_compile.triton('triton_per_fused_mean_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_28(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 2*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ge/cgepemld4bm7vkqfglzmxi2ef2xmh2adz5oikfy7aqzbvwb3dgi6.py
# Topologically Sorted Source Nodes: [mean_3], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_3 => mean_3
# Graph fragment:
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_29, [2, 3], True), kwargs = {})
triton_red_fused_mean_29 = async_compile.triton('triton_red_fused_mean_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_29(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b2/cb2fox44rt6ogiihra7ub3gdnco2f3lmglgwkfewnxtg2eggcedr.py
# Topologically Sorted Source Nodes: [val, mean_1, mean_2, mean_3, mean_4, val_1, val_2, val_3, val_4], Original ATen: [aten.mean, aten.add]
# Source node to ATen node mapping:
#   mean_1 => mean_1
#   mean_2 => mean_2
#   mean_3 => mean_3
#   mean_4 => mean_4
#   val => mean
#   val_1 => add_10
#   val_2 => add_11
#   val_3 => add_12
#   val_4 => add_13
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_26, [2, 3], True), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_27, [2, 3], True), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_28, [2, 3], True), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_29, [2, 3], True), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_30, [2, 3], True), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mean_2), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mean_3), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %mean_4), kwargs = {})
triton_per_fused_add_mean_30 = async_compile.triton('triton_per_fused_add_mean_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 4
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp4 = tl.load(in_out_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
    tmp5 = 262144.0
    tmp6 = tmp4 / tmp5
    tmp8 = 65536.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = 16384.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp10 + tmp13
    tmp16 = 4096.0
    tmp17 = tmp15 / tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 1024.0
    tmp20 = tmp3 / tmp19
    tmp21 = tmp18 + tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (4, 3, 512, 512), (786432, 262144, 512, 1))
    assert_size_stride(primals_3, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_4, (4, 3, 512, 512), (786432, 262144, 512, 1))
    assert_size_stride(primals_5, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_32, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_33, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_34, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_35, (1, 512, 1, 1), (512, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 3, 512, 512), (786432, 1, 1536, 3), torch.float32)
        buf43 = empty_strided_cuda((4, 3, 512, 512), (786432, 1, 1536, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input], Original ATen: [aten.sub, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sub_0.run(primals_2, primals_1, primals_3, primals_4, buf0, buf43, 12, 262144, grid=grid(12, 262144), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        del primals_4
        buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        buf44 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_31], Original ATen: [aten.sub, aten.div, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_sub_1.run(primals_5, buf1, buf44, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1], Original ATen: [aten.sub, aten.div, aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 512, 512), (16777216, 1, 32768, 64))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_relu_sub_2.run(buf3, primals_6, 67108864, grid=grid(67108864), stream=stream0)
        buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        buf47 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_31, input_32, input_33], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_relu_sub_3.run(primals_7, buf4, buf47, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 512, 512), (16777216, 1, 32768, 64))
        del buf3
        del buf4
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31], Original ATen: [aten.sub, aten.div, aten.convolution]
        buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 64, 512, 512), (16777216, 1, 32768, 64))
        del buf43
        del buf44
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_relu_sub_2.run(buf46, primals_6, 67108864, grid=grid(67108864), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        buf48 = extern_kernels.convolution(buf46, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 64, 512, 512), (16777216, 1, 32768, 64))
        del buf47
        buf6 = buf5; del buf5  # reuse
        buf49 = buf48; del buf48  # reuse
        buf88 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_31, input_32, input_33, input_34, pow_1, sum_1, norm_factor, add, truediv_2, pow_2, sum_2, norm_factor_1, add_1, truediv_3, sub_2, pow_3], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.pow, aten.sum, aten.sqrt, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_pow_relu_sqrt_sub_sum_4.run(buf6, buf49, primals_8, buf88, 1048576, 64, grid=grid(1048576), stream=stream0)
        del primals_8
        buf7 = empty_strided_cuda((4, 64, 256, 256), (4194304, 1, 16384, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_5.run(buf6, buf7, 16777216, grid=grid(16777216), stream=stream0)
        del buf6
        buf8 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        buf51 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_31, input_32, input_33, input_34, input_35, input_36], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_6.run(primals_9, buf8, buf51, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 256, 256), (8388608, 1, 32768, 128))
        del buf8
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_7.run(buf10, primals_10, 33554432, grid=grid(33554432), stream=stream0)
        buf11 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf54 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8.run(primals_11, buf11, buf54, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 256, 256), (8388608, 1, 32768, 128))
        del buf10
        del buf11
        buf50 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_5.run(buf49, buf50, 16777216, grid=grid(16777216), stream=stream0)
        del buf49
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 128, 256, 256), (8388608, 1, 32768, 128))
        del buf50
        del buf51
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_7.run(buf53, primals_10, 33554432, grid=grid(33554432), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf55 = extern_kernels.convolution(buf53, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 128, 256, 256), (8388608, 1, 32768, 128))
        del buf54
        buf13 = buf12; del buf12  # reuse
        buf56 = buf55; del buf55  # reuse
        buf91 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, pow_4, sum_3, norm_factor_2, add_2, truediv_4, pow_5, sum_4, norm_factor_3, add_3, truediv_5, sub_3, pow_6], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_9.run(buf13, buf56, primals_12, buf91, 262144, 128, grid=grid(262144), stream=stream0)
        del primals_12
        buf14 = empty_strided_cuda((4, 128, 128, 128), (2097152, 1, 16384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10.run(buf13, buf14, 8388608, grid=grid(8388608), stream=stream0)
        del buf13
        buf15 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf58 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_11.run(primals_13, buf15, buf58, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_12.run(buf17, primals_14, 16777216, grid=grid(16777216), stream=stream0)
        buf18 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        buf61 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_13.run(primals_15, buf18, buf61, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf17
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_12.run(buf20, primals_16, 16777216, grid=grid(16777216), stream=stream0)
        buf21 = buf18; del buf18  # reuse
        buf64 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_13.run(primals_17, buf21, buf64, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf20
        del buf21
        buf57 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10.run(buf56, buf57, 8388608, grid=grid(8388608), stream=stream0)
        del buf56
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf59 = extern_kernels.convolution(buf57, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf57
        del buf58
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_12.run(buf60, primals_14, 16777216, grid=grid(16777216), stream=stream0)
        del primals_14
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf60
        del buf61
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_12.run(buf63, primals_16, 16777216, grid=grid(16777216), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf65 = extern_kernels.convolution(buf63, buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf64
        buf23 = buf22; del buf22  # reuse
        buf66 = buf65; del buf65  # reuse
        buf94 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, pow_7, sum_5, norm_factor_4, add_4, truediv_6, pow_8, sum_6, norm_factor_5, add_5, truediv_7, sub_4, pow_9], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_14.run(buf23, buf66, primals_18, buf94, 65536, 256, grid=grid(65536), stream=stream0)
        del primals_18
        buf24 = empty_strided_cuda((4, 256, 64, 64), (1048576, 1, 16384, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_15.run(buf23, buf24, 4194304, grid=grid(4194304), stream=stream0)
        del buf23
        buf25 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        buf68 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16.run(primals_19, buf25, buf68, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf26 = extern_kernels.convolution(buf24, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        del buf25
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17.run(buf27, primals_20, 8388608, grid=grid(8388608), stream=stream0)
        buf28 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        buf71 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18.run(primals_21, buf28, buf71, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        del buf27
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17.run(buf30, primals_22, 8388608, grid=grid(8388608), stream=stream0)
        buf31 = buf28; del buf28  # reuse
        buf74 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18.run(primals_23, buf31, buf74, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf32 = extern_kernels.convolution(buf30, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        del buf30
        buf67 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_15.run(buf66, buf67, 4194304, grid=grid(4194304), stream=stream0)
        del buf66
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf69 = extern_kernels.convolution(buf67, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        del buf67
        del buf68
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17.run(buf70, primals_20, 8388608, grid=grid(8388608), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf72 = extern_kernels.convolution(buf70, buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        del buf70
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17.run(buf73, primals_22, 8388608, grid=grid(8388608), stream=stream0)
        del primals_22
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf75 = extern_kernels.convolution(buf73, buf74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        buf33 = buf32; del buf32  # reuse
        buf76 = buf75; del buf75  # reuse
        buf97 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, pow_10, sum_7, norm_factor_6, add_6, truediv_8, pow_11, sum_8, norm_factor_7, add_7, truediv_9, sub_5, pow_12], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_19.run(buf33, buf76, primals_24, buf97, 16384, 512, grid=grid(16384), stream=stream0)
        del primals_24
        buf34 = empty_strided_cuda((4, 512, 32, 32), (524288, 1, 16384, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20.run(buf33, buf34, 2097152, grid=grid(2097152), stream=stream0)
        del buf33
        buf35 = buf74; del buf74  # reuse
        buf78 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18.run(primals_25, buf35, buf78, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf36 = extern_kernels.convolution(buf34, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf34
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_21.run(buf37, primals_26, 2097152, grid=grid(2097152), stream=stream0)
        buf38 = buf35; del buf35  # reuse
        buf81 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18.run(primals_27, buf38, buf81, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf39 = extern_kernels.convolution(buf37, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf37
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_21.run(buf40, primals_28, 2097152, grid=grid(2097152), stream=stream0)
        buf41 = buf38; del buf38  # reuse
        buf84 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18.run(primals_29, buf41, buf84, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [sub, in0_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf42 = extern_kernels.convolution(buf40, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf41
        buf77 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20.run(buf76, buf77, 2097152, grid=grid(2097152), stream=stream0)
        del buf76
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf79 = extern_kernels.convolution(buf77, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf77
        del buf78
        buf80 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_21.run(buf80, primals_26, 2097152, grid=grid(2097152), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf82 = extern_kernels.convolution(buf80, buf81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf80
        del buf81
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_21.run(buf83, primals_28, 2097152, grid=grid(2097152), stream=stream0)
        del primals_28
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf85 = extern_kernels.convolution(buf83, buf84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf83
        del buf84
        buf100 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, pow_13, sum_9, norm_factor_8, add_8, truediv_10, pow_14, sum_10, norm_factor_9, add_9, truediv_11, sub_6, pow_15], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_22.run(buf100, primals_30, buf85, 4096, 512, grid=grid(4096), stream=stream0)
        del buf85
        del primals_30
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf88, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 1, 512, 512), (262144, 1, 512, 1))
        buf102 = empty_strided_cuda((4, 1, 1, 1, 32), (32, 128, 128, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [val], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_23.run(buf101, buf102, 128, 8192, grid=grid(128), stream=stream0)
        del buf101
        buf103 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [val], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_24.run(buf102, buf103, 4, 32, grid=grid(4), stream=stream0)
        del buf102
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf91, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 1, 256, 256), (65536, 1, 256, 1))
        buf105 = empty_strided_cuda((4, 1, 1, 1, 8), (8, 32, 32, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_25.run(buf104, buf105, 32, 8192, grid=grid(32), stream=stream0)
        del buf104
        buf106 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_26.run(buf105, buf106, 4, 8, grid=grid(4), stream=stream0)
        del buf105
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf94, primals_33, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 1, 128, 128), (16384, 1, 128, 1))
        buf108 = empty_strided_cuda((4, 1, 1, 1, 2), (2, 8, 8, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mean_2], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_27.run(buf107, buf108, 8, 8192, grid=grid(8), stream=stream0)
        del buf107
        buf109 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mean_2], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_28.run(buf108, buf109, 4, 2, grid=grid(4), stream=stream0)
        del buf108
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf97, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 1, 64, 64), (4096, 1, 64, 1))
        buf111 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mean_3], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_29.run(buf110, buf111, 4, 4096, grid=grid(4), stream=stream0)
        del buf110
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf100, primals_35, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 1, 32, 32), (1024, 1, 32, 1))
        buf114 = reinterpret_tensor(buf103, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [val, mean_1, mean_2, mean_3, mean_4, val_1, val_2, val_3, val_4], Original ATen: [aten.mean, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_30.run(buf114, buf112, buf106, buf109, buf111, 4, 1024, grid=grid(4), stream=stream0)
        del buf106
        del buf109
        del buf111
        del buf112
    return (buf114, primals_31, primals_32, primals_33, primals_34, primals_35, buf88, buf91, buf94, buf97, buf100, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 512, 512), (786432, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 3, 512, 512), (786432, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
