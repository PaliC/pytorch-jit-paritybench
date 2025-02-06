# AOT ID: ['6_inference']
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


# kernel path: inductor_cache/4u/c4uphk3fqlj6q7n67bpi7eytulvyt7qomgbbaq7oo22qmpn4mxtm.py
# Topologically Sorted Source Nodes: [ones_2], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   ones_2 => full_default_10
# Graph fragment:
#   %full_default_10 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 128], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_ones_0 = async_compile.triton('triton_poi_fused_ones_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/ae/caezwt3dazloz6k5qlrlf2btuoq3t2jffhdyjxdg2kdfn4gpc6oh.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   sub_1 => sub_1
#   x => div
#   x_1 => convolution
#   x_2 => relu
#   x_3 => convolution_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_4 => relu_1
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_1 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_1(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/lx/clxoagskmjx5qordpcuqhf5cmcss3cct52b3w7porwml3uigcntv.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   sub_1 => sub_1
#   x => div
#   x_1 => convolution
#   x_2 => relu
#   x_3 => convolution_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_4 => relu_1
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_2 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_2(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/rr/crrkolhzwcxnb7fcwtjvvn2euallcs5uu2tul4tymmeil3psm7ci.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, sub_1, x_36, x_37, x_38, x_39], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   sub => sub
#   sub_1 => sub_1
#   x => div
#   x_1 => convolution
#   x_2 => relu
#   x_3 => convolution_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_relu_sub_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: inductor_cache/it/citnzpf6ukmbpkqrzwbjhz467f2dfgroknt742n6byzqdy2totcn.py
# Topologically Sorted Source Nodes: [sub, x, x_1, sub_1, x_36, x_37], Original ATen: [aten.sub, aten.div, aten.convolution]
# Source node to ATen node mapping:
#   sub => sub
#   sub_1 => sub_1
#   x => div
#   x_1 => convolution
#   x_36 => div_1
#   x_37 => convolution_16
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_sub_4 = async_compile.triton('triton_poi_fused_convolution_div_sub_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_sub_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_sub_4(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/hv/chvlqtgbvtew2pkavtfmpmi2no4jyfczetydksthpgitvsj363ma.py
# Topologically Sorted Source Nodes: [sub, x], Original ATen: [aten.sub, aten.div]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
triton_poi_fused_div_sub_5 = async_compile.triton('triton_poi_fused_div_sub_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sub_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_sub_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp4, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/kg/ckg3urqn2dd4lzcaeoagf5gwyuvefbhaoswn2vlc24b7t6e7iqun.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_2 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_div_relu_sub_6 = async_compile.triton('triton_poi_fused_convolution_div_relu_sub_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_relu_sub_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_relu_sub_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: inductor_cache/vw/cvw3al4oi5lzxpvwkqmaimlwmwbsg6ra2yoxuflcvapxe6gx5zwq.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_2 => relu
#   x_3 => convolution_1
#   x_4 => relu_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
triton_poi_fused_convolution_div_relu_sub_7 = async_compile.triton('triton_poi_fused_convolution_div_relu_sub_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_relu_sub_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_relu_sub_7(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 262144*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp4, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/oe/coexrasmurz46jvg3thc4m26acaei6nrroxm5bcg5mrtl432vbj3.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_2 => relu
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 32)
    x3 = xindex // 32
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (2*x2 + 128*x3 + 4096*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 128*x3 + 4096*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x2 + 128*x3 + 4096*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x2 + 128*x3 + 4096*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + 64*x5 + 65536*y1), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/4o/c4o7i3qjuppjqidahaxm5bzhq5xsldnnx7gdtefwxw2pfhdofq5r.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_2 => relu
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: inductor_cache/yk/cykebbth7kjxomqdyv637d4zebpzjwyo35iom6nuk35ytcsomf5h.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_2 => relu
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 131072*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 1024*y3), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/7y/c7yplctydqvomwece7vi6vev2reakycwxse7sfybcqvpusutldeu.py
# Topologically Sorted Source Nodes: [repeat_2, mul_21, mul_22], Original ATen: [aten.repeat, aten.mul]
# Source node to ATen node mapping:
#   mul_21 => mul_24
#   mul_22 => mul_25
#   repeat_2 => repeat_2
# Graph fragment:
#   %repeat_2 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%view_9, [4, 1, 1]), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_14, %repeat_2), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_14, %repeat_2), kwargs = {})
triton_poi_fused_mul_repeat_11 = async_compile.triton('triton_poi_fused_mul_repeat_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_repeat_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_repeat_11(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 128) % 128)
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = x1
    tmp2 = x0
    tmp3 = tmp1 == tmp2
    tmp4 = 1.0
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/ya/cyavguuiwrhd25nm3uvga4imk3csx5sr4eadqg6xvitmio4xy6ew.py
# Topologically Sorted Source Nodes: [exp_2, add_8, mul_23, dcov_8, dcov_9, dcov_10, add_9, dcov_11], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
# Source node to ATen node mapping:
#   add_8 => add_8
#   add_9 => add_9
#   dcov_10 => mul_27
#   dcov_11 => sqrt_3
#   dcov_8 => sub_8
#   dcov_9 => clamp_min_2
#   exp_2 => full_default_11
#   mul_23 => mul_26
# Graph fragment:
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 6.103515625e-05), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_15, %bmm_16), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_14, 2), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %mul_26), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_8, 0.0), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_11, %clamp_min_2), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, 1e-05), kwargs = {})
#   %sqrt_3 : [num_users=4] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_9,), kwargs = {})
triton_poi_fused_add_clamp_exp_mul_sqrt_sub_12 = async_compile.triton('triton_poi_fused_add_clamp_exp_mul_sqrt_sub_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_exp_mul_sqrt_sub_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_exp_mul_sqrt_sub_12(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 6.103515625e-05
    tmp10 = tmp9 * tmp8
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tl.store(in_out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/qt/cqttxpins755lmba6tmii3pj7otftg3ydsrszivx2c7m5vofjaru.py
# Topologically Sorted Source Nodes: [sub_1, x_36], Original ATen: [aten.sub, aten.div]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
triton_poi_fused_div_sub_13 = async_compile.triton('triton_poi_fused_div_sub_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sub_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_sub_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tl.store(out_ptr0 + (y0 + 3*x2 + 786432*y1), tmp4, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/l4/cl4fabuq6fs3hvjklqmspp23uc4cidamix2dan7tmfvziu3o6f6r.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
triton_poi_fused_convolution_div_relu_sub_14 = async_compile.triton('triton_poi_fused_convolution_div_relu_sub_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_relu_sub_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_relu_sub_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/nc/cnc2kqqwvia5gdxbjjyl44vjfwzilknjvxg26ecvyrrj6ndx4cxu.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
triton_poi_fused_convolution_div_relu_sub_15 = async_compile.triton('triton_poi_fused_convolution_div_relu_sub_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 262144}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_relu_sub_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_relu_sub_15(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 262144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 16777216*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 262144*y3), tmp4, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/pq/cpqgx3pqeahxytpbugniotlx23eddvrfgqdj7j6uh47lplu3t6xy.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = (xindex % 256)
    x3 = xindex // 256
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (2*x2 + 1024*x3 + 262144*y4), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 1024*x3 + 262144*y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (512 + 2*x2 + 1024*x3 + 262144*y4), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (513 + 2*x2 + 1024*x3 + 262144*y4), ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + 64*x5 + 4194304*y1), tmp6, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/kd/ckd2ryfdtkv44wiawkdredaf6zz4dlmxkchlcugqhh7vsxkeoa2t.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/f7/cf7y4k57xzogbyfip75pq7x3nb6vyyho6z3oellxx3zu3ulxetws.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 65536}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 8388608*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 65536*y3), tmp4, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/c4/cc4cxnivmqnkcuni2wbcuxq5h6gsvxt3dhyelkngo5rb6fen6wm4.py
# Topologically Sorted Source Nodes: [mul_25, sub_9, mul_26, sub_10, mul_27, dcdm_2, mul_33, sub_12, mul_34, sub_13, mul_35, dcdm_3, mul_36, Gamma_XY_1, mul_37, Gamma_XX_1, mul_38, Gamma_YY_1], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum]
# Source node to ATen node mapping:
#   Gamma_XX_1 => sum_5
#   Gamma_XY_1 => sum_4
#   Gamma_YY_1 => sum_6
#   dcdm_2 => add_10
#   dcdm_3 => add_13
#   mul_25 => mul_28
#   mul_26 => mul_29
#   mul_27 => mul_30
#   mul_33 => mul_37
#   mul_34 => mul_38
#   mul_35 => mul_39
#   mul_36 => mul_40
#   mul_37 => mul_41
#   mul_38 => mul_42
#   sub_10 => sub_10
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sub_9 => sub_9
# Graph fragment:
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_17, 0.0078125), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_3, %mul_28), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_18, 0.0078125), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_9, %mul_29), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_20, 6.103515625e-05), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_10, %mul_30), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_24, 0.0078125), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_4, %mul_37), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_25, 0.0078125), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %mul_38), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_27, 6.103515625e-05), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_13, %mul_39), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %add_13), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [1, 2]), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %add_10), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_41, [1, 2]), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %add_13), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_42, [1, 2]), kwargs = {})
triton_red_fused_add_mul_sub_sum_19 = async_compile.triton('triton_red_fused_add_mul_sub_sum_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sub_sum_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_sub_sum_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr5 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr6 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr7 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0078125
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 - tmp3
        tmp6 = tmp5 * tmp2
        tmp7 = tmp4 - tmp6
        tmp9 = 6.103515625e-05
        tmp10 = tmp8 * tmp9
        tmp11 = tmp7 + tmp10
        tmp14 = tmp13 * tmp2
        tmp15 = tmp12 - tmp14
        tmp17 = tmp16 * tmp2
        tmp18 = tmp15 - tmp17
        tmp20 = tmp19 * tmp9
        tmp21 = tmp18 + tmp20
        tmp22 = tmp11 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp26 = tmp11 * tmp11
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tmp30 = tmp21 * tmp21
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp24, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
    tl.store(out_ptr2 + (x0), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7i/c7i667oq44lt437lh7ijd2fawdbunfuyskqyecrkmgjsq6sf3bka.py
# Topologically Sorted Source Nodes: [mul_25, sub_9, mul_26, sub_10, mul_27, dcdm_2, mul_33, sub_12, mul_34, sub_13, mul_35, dcdm_3, mul_36, Gamma_XY_1, mul_37, Gamma_XX_1, mul_38, Gamma_YY_1, dc_scores], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum, aten.stack]
# Source node to ATen node mapping:
#   Gamma_XX_1 => sum_5
#   Gamma_XY_1 => sum_4
#   Gamma_YY_1 => sum_6
#   dc_scores => cat
#   dcdm_2 => add_10
#   dcdm_3 => add_13
#   mul_25 => mul_28
#   mul_26 => mul_29
#   mul_27 => mul_30
#   mul_33 => mul_37
#   mul_34 => mul_38
#   mul_35 => mul_39
#   mul_36 => mul_40
#   mul_37 => mul_41
#   mul_38 => mul_42
#   sub_10 => sub_10
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sub_9 => sub_9
# Graph fragment:
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_17, 0.0078125), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_3, %mul_28), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_18, 0.0078125), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_9, %mul_29), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_20, 6.103515625e-05), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_10, %mul_30), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_24, 0.0078125), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_4, %mul_37), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_25, 0.0078125), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %mul_38), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_27, 6.103515625e-05), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_13, %mul_39), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %add_13), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [1, 2]), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %add_10), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_41, [1, 2]), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %add_13), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_42, [1, 2]), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_10, %unsqueeze_11, %unsqueeze_12, %unsqueeze_13, %unsqueeze_14], 1), kwargs = {})
triton_per_fused_add_mul_stack_sub_sum_20 = async_compile.triton('triton_per_fused_add_mul_stack_sub_sum_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_stack_sub_sum_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_stack_sub_sum_20(in_ptr0, in_ptr1, in_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (r1 + 2*x0), xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + 2*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = 1e-06
    tmp16 = tmp4 + tmp15
    tmp17 = tmp9 * tmp14
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp18 + tmp15
    tmp20 = tmp16 / tmp19
    tl.store(out_ptr3 + (5*x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3d/c3dszbnfxhi6mel6ylwm373o7ssxzdwapx23km72gkryrizfnos4.py
# Topologically Sorted Source Nodes: [ones_4], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   ones_4 => full_default_18
# Graph fragment:
#   %full_default_18 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([4, 256, 256], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_ones_21 = async_compile.triton('triton_poi_fused_ones_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/fb/cfbfjntq4hakpxqf6minwds45mpckhrqfpwwt4a5gz3ac6kbhipf.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   sub_1 => sub_1
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_2 => relu
#   x_3 => convolution_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_4 => relu_1
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_22 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_22(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/qz/cqzqjh3cssqlrg3q6nteutfqekzvtg6ienxjosjp2iqpnhkg5u66.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_2 => relu
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_23 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 16)
    x3 = xindex // 16
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (2*x2 + 64*x3 + 1024*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 64*x3 + 1024*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x2 + 64*x3 + 1024*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x2 + 64*x3 + 1024*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + 128*x5 + 32768*y1), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/m5/cm5eye3svuxtcojajpyhrwjumbwskmumotob4poekvcw4loxbxhw.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_2 => relu
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_24 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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


# kernel path: inductor_cache/yu/cyuwsgmljehmqabc46hh6ayd7hjjpfuxeghzw2fevssdtnp5frs7.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   sub_1 => sub_1
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_2 => relu
#   x_3 => convolution_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_4 => relu_1
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/iq/ciqrgfh3kg3cweeo4nonbgz2ovljh67er5v5mtavgsx3fbu7732h.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_14 => relu_5
#   x_15 => convolution_6
#   x_16 => relu_6
#   x_17 => convolution_7
#   x_18 => relu_7
#   x_2 => relu
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_26 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_26(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 65536*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 256*y3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/de/cderbz5kj3xmpm4orwj22mb6gtcjwcyoq52wsa3jbwxk4cmmjp3d.py
# Topologically Sorted Source Nodes: [repeat_4, mul_41, mul_42], Original ATen: [aten.repeat, aten.mul]
# Source node to ATen node mapping:
#   mul_41 => mul_46
#   mul_42 => mul_47
#   repeat_4 => repeat_4
# Graph fragment:
#   %repeat_4 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%view_17, [4, 1, 1]), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_28, %repeat_4), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_28, %repeat_4), kwargs = {})
triton_poi_fused_mul_repeat_27 = async_compile.triton('triton_poi_fused_mul_repeat_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_repeat_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_repeat_27(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 256)
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = x1
    tmp2 = x0
    tmp3 = tmp1 == tmp2
    tmp4 = 1.0
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/oe/coeu43v4eqkl2bcv5x5j22rh2m66ut2qyz5blbe62bbbozwoonc5.py
# Topologically Sorted Source Nodes: [exp_4, add_16, mul_43, dcov_16, dcov_17, dcov_18, add_17, dcov_19], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
# Source node to ATen node mapping:
#   add_16 => add_16
#   add_17 => add_17
#   dcov_16 => sub_14
#   dcov_17 => clamp_min_4
#   dcov_18 => mul_49
#   dcov_19 => sqrt_6
#   exp_4 => full_default_19
#   mul_43 => mul_48
# Graph fragment:
#   %full_default_19 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.5258788153005298e-05), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_29, %bmm_30), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_28, 2), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %mul_48), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_14, 0.0), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_19, %clamp_min_4), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, 1e-05), kwargs = {})
#   %sqrt_6 : [num_users=4] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_17,), kwargs = {})
triton_poi_fused_add_clamp_exp_mul_sqrt_sub_28 = async_compile.triton('triton_poi_fused_add_clamp_exp_mul_sqrt_sub_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_exp_mul_sqrt_sub_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_exp_mul_sqrt_sub_28(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1.5258788153005298e-05
    tmp10 = tmp9 * tmp8
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tl.store(in_out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/gf/cgfz7srtyuyeeomnsa4yhd6zqdc34f2pgh6onmkvq6qkwenoz6kc.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_29 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16384}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = (xindex % 128)
    x3 = xindex // 128
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (2*x2 + 512*x3 + 65536*y4), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 512*x3 + 65536*y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (256 + 2*x2 + 512*x3 + 65536*y4), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (257 + 2*x2 + 512*x3 + 65536*y4), ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + 128*x5 + 2097152*y1), tmp6, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/wf/cwfcn7tcxurfr72bvpgseejdtfxdfo5kts5zmljjdif5labx6hal.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_30 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_30(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/j3/cj3tlvpfpun2rpl557cio53halkqpeo3ggqidzc7xtmy37zlkldj.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_50 => relu_20
#   x_51 => convolution_22
#   x_52 => relu_21
#   x_53 => convolution_23
#   x_54 => relu_22
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 4194304*y1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 16384*y3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/ez/cezmhvk5s7xoyhie6dxetwtwoe7cjgjsxcwb2w6er46rg6nvod7x.py
# Topologically Sorted Source Nodes: [mul_45, sub_15, mul_46, sub_16, mul_47, dcdm_4, mul_53, sub_18, mul_54, sub_19, mul_55, dcdm_5, mul_56, Gamma_XY_2, mul_57, Gamma_XX_2, mul_58, Gamma_YY_2], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum]
# Source node to ATen node mapping:
#   Gamma_XX_2 => sum_8
#   Gamma_XY_2 => sum_7
#   Gamma_YY_2 => sum_9
#   dcdm_4 => add_18
#   dcdm_5 => add_21
#   mul_45 => mul_50
#   mul_46 => mul_51
#   mul_47 => mul_52
#   mul_53 => mul_59
#   mul_54 => mul_60
#   mul_55 => mul_61
#   mul_56 => mul_62
#   mul_57 => mul_63
#   mul_58 => mul_64
#   sub_15 => sub_15
#   sub_16 => sub_16
#   sub_18 => sub_18
#   sub_19 => sub_19
# Graph fragment:
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_31, 0.00390625), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_6, %mul_50), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_32, 0.00390625), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_15, %mul_51), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_34, 1.52587890625e-05), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_16, %mul_52), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_38, 0.00390625), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_7, %mul_59), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_39, 0.00390625), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_18, %mul_60), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_41, 1.52587890625e-05), kwargs = {})
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_19, %mul_61), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %add_21), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_62, [1, 2]), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %add_18), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_63, [1, 2]), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %add_21), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_64, [1, 2]), kwargs = {})
triton_red_fused_add_mul_sub_sum_32 = async_compile.triton('triton_red_fused_add_mul_sub_sum_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sub_sum_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_sub_sum_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr5 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr6 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr7 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.00390625
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 - tmp3
        tmp6 = tmp5 * tmp2
        tmp7 = tmp4 - tmp6
        tmp9 = 1.52587890625e-05
        tmp10 = tmp8 * tmp9
        tmp11 = tmp7 + tmp10
        tmp14 = tmp13 * tmp2
        tmp15 = tmp12 - tmp14
        tmp17 = tmp16 * tmp2
        tmp18 = tmp15 - tmp17
        tmp20 = tmp19 * tmp9
        tmp21 = tmp18 + tmp20
        tmp22 = tmp11 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp26 = tmp11 * tmp11
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tmp30 = tmp21 * tmp21
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp24, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
    tl.store(out_ptr2 + (x0), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sb/csbuxprofcvghtgzfrp4xmdzagkjd3mcy2dvptvxj6vyh3y2ajxb.py
# Topologically Sorted Source Nodes: [mul_45, sub_15, mul_46, sub_16, mul_47, dcdm_4, mul_53, sub_18, mul_54, sub_19, mul_55, dcdm_5, mul_56, Gamma_XY_2, mul_57, Gamma_XX_2, mul_58, Gamma_YY_2, dc_scores], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum, aten.stack]
# Source node to ATen node mapping:
#   Gamma_XX_2 => sum_8
#   Gamma_XY_2 => sum_7
#   Gamma_YY_2 => sum_9
#   dc_scores => cat
#   dcdm_4 => add_18
#   dcdm_5 => add_21
#   mul_45 => mul_50
#   mul_46 => mul_51
#   mul_47 => mul_52
#   mul_53 => mul_59
#   mul_54 => mul_60
#   mul_55 => mul_61
#   mul_56 => mul_62
#   mul_57 => mul_63
#   mul_58 => mul_64
#   sub_15 => sub_15
#   sub_16 => sub_16
#   sub_18 => sub_18
#   sub_19 => sub_19
# Graph fragment:
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_31, 0.00390625), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_6, %mul_50), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_32, 0.00390625), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_15, %mul_51), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_34, 1.52587890625e-05), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_16, %mul_52), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_38, 0.00390625), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_7, %mul_59), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_39, 0.00390625), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_18, %mul_60), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_41, 1.52587890625e-05), kwargs = {})
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_19, %mul_61), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %add_21), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_62, [1, 2]), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %add_18), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_63, [1, 2]), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %add_21), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_64, [1, 2]), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_10, %unsqueeze_11, %unsqueeze_12, %unsqueeze_13, %unsqueeze_14], 1), kwargs = {})
triton_per_fused_add_mul_stack_sub_sum_33 = async_compile.triton('triton_per_fused_add_mul_stack_sub_sum_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_stack_sub_sum_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_stack_sub_sum_33(in_ptr0, in_ptr1, in_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (r1 + 8*x0), xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + 8*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = 1e-06
    tmp16 = tmp4 + tmp15
    tmp17 = tmp9 * tmp14
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp18 + tmp15
    tmp20 = tmp16 / tmp19
    tl.store(out_ptr3 + (5*x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gv/cgvq2m3a77v6ovwh3kuhctq7izkxf2j2z7obpp7jefmi6lhip4bx.py
# Topologically Sorted Source Nodes: [ones_6], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   ones_6 => full_default_26
# Graph fragment:
#   %full_default_26 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([4, 512, 512], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_ones_34 = async_compile.triton('triton_poi_fused_ones_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_34(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/dd/cdd2pjizwfvgzygimhruablz7j3fyjg2hmwcvu2pvoh74hr6ugvk.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_14 => relu_5
#   x_15 => convolution_6
#   x_16 => relu_6
#   x_17 => convolution_7
#   x_18 => relu_7
#   x_19 => _low_memory_max_pool2d_with_offsets_2
#   x_2 => relu
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_35 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 8)
    x3 = xindex // 8
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (2*x2 + 32*x3 + 256*y4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 32*x3 + 256*y4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x2 + 32*x3 + 256*y4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x2 + 32*x3 + 256*y4), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + 256*x5 + 16384*y1), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gv/cgvliqgeblmyvvq4txz4cidrumc52utjlek5xhrszvuruudcsndk.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   sub_1 => sub_1
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_14 => relu_5
#   x_15 => convolution_6
#   x_16 => relu_6
#   x_17 => convolution_7
#   x_18 => relu_7
#   x_19 => _low_memory_max_pool2d_with_offsets_2
#   x_2 => relu
#   x_20 => convolution_8
#   x_3 => convolution_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_4 => relu_1
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_50 => relu_20
#   x_51 => convolution_22
#   x_52 => relu_21
#   x_53 => convolution_23
#   x_54 => relu_22
#   x_55 => _low_memory_max_pool2d_with_offsets_6
#   x_56 => convolution_24
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_36 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_36(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/we/cwetxl3zgnau4mu3vmiiffnls6utrqt6rrtwkca7sozek4ms7sem.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_14 => relu_5
#   x_15 => convolution_6
#   x_16 => relu_6
#   x_17 => convolution_7
#   x_18 => relu_7
#   x_19 => _low_memory_max_pool2d_with_offsets_2
#   x_2 => relu
#   x_20 => convolution_8
#   x_21 => relu_8
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_37 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_37(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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


# kernel path: inductor_cache/dr/cdryd4bhxduj7a5d4caqlfiuoezkuunlirso453zffifkusvmtwm.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   sub_1 => sub_1
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_14 => relu_5
#   x_15 => convolution_6
#   x_16 => relu_6
#   x_17 => convolution_7
#   x_18 => relu_7
#   x_19 => _low_memory_max_pool2d_with_offsets_2
#   x_2 => relu
#   x_20 => convolution_8
#   x_21 => relu_8
#   x_22 => convolution_9
#   x_3 => convolution_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_4 => relu_1
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_50 => relu_20
#   x_51 => convolution_22
#   x_52 => relu_21
#   x_53 => convolution_23
#   x_54 => relu_22
#   x_55 => _low_memory_max_pool2d_with_offsets_6
#   x_56 => convolution_24
#   x_57 => relu_23
#   x_58 => convolution_25
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/xw/cxwqnu5wptxcgkurc4idyth6wqeasp45qtklkdo4usbs7pcm3i75.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_14 => relu_5
#   x_15 => convolution_6
#   x_16 => relu_6
#   x_17 => convolution_7
#   x_18 => relu_7
#   x_19 => _low_memory_max_pool2d_with_offsets_2
#   x_2 => relu
#   x_20 => convolution_8
#   x_21 => relu_8
#   x_22 => convolution_9
#   x_23 => relu_9
#   x_24 => convolution_10
#   x_25 => relu_10
#   x_26 => convolution_11
#   x_27 => relu_11
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg24_1, %arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_39 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_39(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 32768*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 64*y3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jq/cjq3tdkgwqp6lmb7rpnpk25pjyyte23l4n5y5swniymbkqwa4ek7.py
# Topologically Sorted Source Nodes: [repeat_6, mul_61, mul_62], Original ATen: [aten.repeat, aten.mul]
# Source node to ATen node mapping:
#   mul_61 => mul_68
#   mul_62 => mul_69
#   repeat_6 => repeat_6
# Graph fragment:
#   %repeat_6 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%view_25, [4, 1, 1]), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_42, %repeat_6), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_42, %repeat_6), kwargs = {})
triton_poi_fused_mul_repeat_40 = async_compile.triton('triton_poi_fused_mul_repeat_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_repeat_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_repeat_40(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 512)
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = x1
    tmp2 = x0
    tmp3 = tmp1 == tmp2
    tmp4 = 1.0
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/i5/ci5b6abj7wh464v2wb27hjpwbuia43p7k2ik7rmehfeonlsivou2.py
# Topologically Sorted Source Nodes: [exp_6, add_24, mul_63, dcov_24, dcov_25, dcov_26, add_25, dcov_27], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
# Source node to ATen node mapping:
#   add_24 => add_24
#   add_25 => add_25
#   dcov_24 => sub_20
#   dcov_25 => clamp_min_6
#   dcov_26 => mul_71
#   dcov_27 => sqrt_9
#   exp_6 => full_default_27
#   mul_63 => mul_70
# Graph fragment:
#   %full_default_27 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 3.8146970382513246e-06), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_43, %bmm_44), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_42, 2), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %mul_70), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_20, 0.0), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_27, %clamp_min_6), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, 1e-05), kwargs = {})
#   %sqrt_9 : [num_users=4] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_25,), kwargs = {})
triton_poi_fused_add_clamp_exp_mul_sqrt_sub_41 = async_compile.triton('triton_poi_fused_add_clamp_exp_mul_sqrt_sub_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_exp_mul_sqrt_sub_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_exp_mul_sqrt_sub_41(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 3.8146970382513246e-06
    tmp10 = tmp9 * tmp8
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tl.store(in_out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/bu/cbud6knn2y57dh6roes5adenyj2xgidzxpiespagnlfnac7v5ozn.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_50 => relu_20
#   x_51 => convolution_22
#   x_52 => relu_21
#   x_53 => convolution_23
#   x_54 => relu_22
#   x_55 => _low_memory_max_pool2d_with_offsets_6
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_42 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_42(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = (xindex % 64)
    x3 = xindex // 64
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (2*x2 + 256*x3 + 16384*y4), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 256*x3 + 16384*y4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + 2*x2 + 256*x3 + 16384*y4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (129 + 2*x2 + 256*x3 + 16384*y4), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + 256*x5 + 1048576*y1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/3y/c3y7rwqc3zy2j325hdmu3cbeugafb4sdgmoop5zz3gj73ndc6ksk.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_50 => relu_20
#   x_51 => convolution_22
#   x_52 => relu_21
#   x_53 => convolution_23
#   x_54 => relu_22
#   x_55 => _low_memory_max_pool2d_with_offsets_6
#   x_56 => convolution_24
#   x_57 => relu_23
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_43 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_43(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/3v/c3vmfrr47ib3in7qtnqueopr5e3r33dfnlpystxdgnh2q4pz24st.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_50 => relu_20
#   x_51 => convolution_22
#   x_52 => relu_21
#   x_53 => convolution_23
#   x_54 => relu_22
#   x_55 => _low_memory_max_pool2d_with_offsets_6
#   x_56 => convolution_24
#   x_57 => relu_23
#   x_58 => convolution_25
#   x_59 => relu_24
#   x_60 => convolution_26
#   x_61 => relu_25
#   x_62 => convolution_27
#   x_63 => relu_26
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_24 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg24_1, %arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_26,), kwargs = {})
#   %convolution_27 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_26 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_27,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_44 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_44(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 2097152*y1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/2y/c2ybj64mlp5tjh6ti234lxsypxiza2du3cwgmx5trawkkmxzeole.py
# Topologically Sorted Source Nodes: [mul_65, sub_21, mul_66, sub_22, mul_67, dcdm_6, mul_73, sub_24, mul_74, sub_25, mul_75, dcdm_7, mul_76, Gamma_XY_3, mul_77, Gamma_XX_3, mul_78, Gamma_YY_3], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum]
# Source node to ATen node mapping:
#   Gamma_XX_3 => sum_11
#   Gamma_XY_3 => sum_10
#   Gamma_YY_3 => sum_12
#   dcdm_6 => add_26
#   dcdm_7 => add_29
#   mul_65 => mul_72
#   mul_66 => mul_73
#   mul_67 => mul_74
#   mul_73 => mul_81
#   mul_74 => mul_82
#   mul_75 => mul_83
#   mul_76 => mul_84
#   mul_77 => mul_85
#   mul_78 => mul_86
#   sub_21 => sub_21
#   sub_22 => sub_22
#   sub_24 => sub_24
#   sub_25 => sub_25
# Graph fragment:
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_45, 0.001953125), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_9, %mul_72), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_46, 0.001953125), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_21, %mul_73), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_48, 3.814697265625e-06), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_22, %mul_74), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_52, 0.001953125), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_10, %mul_81), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_53, 0.001953125), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_24, %mul_82), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_55, 3.814697265625e-06), kwargs = {})
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_25, %mul_83), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %add_29), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_84, [1, 2]), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %add_26), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_85, [1, 2]), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %add_29), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_86, [1, 2]), kwargs = {})
triton_red_fused_add_mul_sub_sum_45 = async_compile.triton('triton_red_fused_add_mul_sub_sum_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sub_sum_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_sub_sum_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr5 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr6 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr7 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.001953125
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 - tmp3
        tmp6 = tmp5 * tmp2
        tmp7 = tmp4 - tmp6
        tmp9 = 3.814697265625e-06
        tmp10 = tmp8 * tmp9
        tmp11 = tmp7 + tmp10
        tmp14 = tmp13 * tmp2
        tmp15 = tmp12 - tmp14
        tmp17 = tmp16 * tmp2
        tmp18 = tmp15 - tmp17
        tmp20 = tmp19 * tmp9
        tmp21 = tmp18 + tmp20
        tmp22 = tmp11 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp26 = tmp11 * tmp11
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tmp30 = tmp21 * tmp21
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp24, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
    tl.store(out_ptr2 + (x0), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ey/cey6iyfm3blcmlde3ozyjijfch4nbn75g24rzqtxem6houm2gyra.py
# Topologically Sorted Source Nodes: [mul_65, sub_21, mul_66, sub_22, mul_67, dcdm_6, mul_73, sub_24, mul_74, sub_25, mul_75, dcdm_7, mul_76, Gamma_XY_3, mul_77, Gamma_XX_3, mul_78, Gamma_YY_3, dc_scores], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum, aten.stack]
# Source node to ATen node mapping:
#   Gamma_XX_3 => sum_11
#   Gamma_XY_3 => sum_10
#   Gamma_YY_3 => sum_12
#   dc_scores => cat
#   dcdm_6 => add_26
#   dcdm_7 => add_29
#   mul_65 => mul_72
#   mul_66 => mul_73
#   mul_67 => mul_74
#   mul_73 => mul_81
#   mul_74 => mul_82
#   mul_75 => mul_83
#   mul_76 => mul_84
#   mul_77 => mul_85
#   mul_78 => mul_86
#   sub_21 => sub_21
#   sub_22 => sub_22
#   sub_24 => sub_24
#   sub_25 => sub_25
# Graph fragment:
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_45, 0.001953125), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_9, %mul_72), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_46, 0.001953125), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_21, %mul_73), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_48, 3.814697265625e-06), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_22, %mul_74), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_52, 0.001953125), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_10, %mul_81), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_53, 0.001953125), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_24, %mul_82), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_55, 3.814697265625e-06), kwargs = {})
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_25, %mul_83), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %add_29), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_84, [1, 2]), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %add_26), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_85, [1, 2]), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %add_29), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_86, [1, 2]), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_10, %unsqueeze_11, %unsqueeze_12, %unsqueeze_13, %unsqueeze_14], 1), kwargs = {})
triton_per_fused_add_mul_stack_sub_sum_46 = async_compile.triton('triton_per_fused_add_mul_stack_sub_sum_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_stack_sub_sum_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_stack_sub_sum_46(in_ptr0, in_ptr1, in_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (r1 + 32*x0), xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = 1e-06
    tmp16 = tmp4 + tmp15
    tmp17 = tmp9 * tmp14
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp18 + tmp15
    tmp20 = tmp16 / tmp19
    tl.store(out_ptr3 + (5*x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m7/cm7fdwguvl4tbclhou7kdty2ysawnkfdgvjywfuzspan2spodll6.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_14 => relu_5
#   x_15 => convolution_6
#   x_16 => relu_6
#   x_17 => convolution_7
#   x_18 => relu_7
#   x_19 => _low_memory_max_pool2d_with_offsets_2
#   x_2 => relu
#   x_20 => convolution_8
#   x_21 => relu_8
#   x_22 => convolution_9
#   x_23 => relu_9
#   x_24 => convolution_10
#   x_25 => relu_10
#   x_26 => convolution_11
#   x_27 => relu_11
#   x_28 => _low_memory_max_pool2d_with_offsets_3
#   x_3 => convolution_1
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg24_1, %arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_47 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 4)
    x3 = xindex // 4
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (2*x2 + 16*x3 + 64*y4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 16*x3 + 64*y4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x2 + 16*x3 + 64*y4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x2 + 16*x3 + 64*y4), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + 512*x5 + 8192*y1), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dc/cdcwjwxtlkqhgilrullqxguy5yoiubhrqtj3etz3qwd7xszqowks.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_14 => relu_5
#   x_15 => convolution_6
#   x_16 => relu_6
#   x_17 => convolution_7
#   x_18 => relu_7
#   x_19 => _low_memory_max_pool2d_with_offsets_2
#   x_2 => relu
#   x_20 => convolution_8
#   x_21 => relu_8
#   x_22 => convolution_9
#   x_23 => relu_9
#   x_24 => convolution_10
#   x_25 => relu_10
#   x_26 => convolution_11
#   x_27 => relu_11
#   x_28 => _low_memory_max_pool2d_with_offsets_3
#   x_29 => convolution_12
#   x_3 => convolution_1
#   x_30 => relu_12
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg24_1, %arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg28_1, %arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_48 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_48(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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


# kernel path: inductor_cache/3w/c3wgjrcacagsfwc5u2ooonww3c3mdfuebwucn6cg5k3yvax6lpln.py
# Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33, x_34, x_35], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
#   x_1 => convolution
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => convolution_4
#   x_12 => relu_4
#   x_13 => convolution_5
#   x_14 => relu_5
#   x_15 => convolution_6
#   x_16 => relu_6
#   x_17 => convolution_7
#   x_18 => relu_7
#   x_19 => _low_memory_max_pool2d_with_offsets_2
#   x_2 => relu
#   x_20 => convolution_8
#   x_21 => relu_8
#   x_22 => convolution_9
#   x_23 => relu_9
#   x_24 => convolution_10
#   x_25 => relu_10
#   x_26 => convolution_11
#   x_27 => relu_11
#   x_28 => _low_memory_max_pool2d_with_offsets_3
#   x_29 => convolution_12
#   x_3 => convolution_1
#   x_30 => relu_12
#   x_31 => convolution_13
#   x_32 => relu_13
#   x_33 => convolution_14
#   x_34 => relu_14
#   x_35 => convolution_15
#   x_4 => relu_1
#   x_5 => _low_memory_max_pool2d_with_offsets
#   x_6 => convolution_2
#   x_7 => relu_2
#   x_8 => convolution_3
#   x_9 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %arg3_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg24_1, %arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg28_1, %arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg30_1, %arg31_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg32_1, %arg33_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg34_1, %arg35_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_49 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_49', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_49(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 8192*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 16*y3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xz/cxzan4jsydudlfxx2x2pktpcbwczrpzjzb65ih6ycn6c454dlzsv.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_50 => relu_20
#   x_51 => convolution_22
#   x_52 => relu_21
#   x_53 => convolution_23
#   x_54 => relu_22
#   x_55 => _low_memory_max_pool2d_with_offsets_6
#   x_56 => convolution_24
#   x_57 => relu_23
#   x_58 => convolution_25
#   x_59 => relu_24
#   x_60 => convolution_26
#   x_61 => relu_25
#   x_62 => convolution_27
#   x_63 => relu_26
#   x_64 => _low_memory_max_pool2d_with_offsets_7
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_24 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg24_1, %arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_26,), kwargs = {})
#   %convolution_27 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_26 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_27,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_7 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_26, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_50 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_50', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_50(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 32)
    x3 = xindex // 32
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (2*x2 + 128*x3 + 4096*y4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 128*x3 + 4096*y4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x2 + 128*x3 + 4096*y4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x2 + 128*x3 + 4096*y4), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + 512*x5 + 524288*y1), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xd/cxdnrwaxqamyw7ilopuzxobmhk3qjevkalgv3arfx3qmx5czzc3p.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_50 => relu_20
#   x_51 => convolution_22
#   x_52 => relu_21
#   x_53 => convolution_23
#   x_54 => relu_22
#   x_55 => _low_memory_max_pool2d_with_offsets_6
#   x_56 => convolution_24
#   x_57 => relu_23
#   x_58 => convolution_25
#   x_59 => relu_24
#   x_60 => convolution_26
#   x_61 => relu_25
#   x_62 => convolution_27
#   x_63 => relu_26
#   x_64 => _low_memory_max_pool2d_with_offsets_7
#   x_65 => convolution_28
#   x_66 => relu_27
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_24 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg24_1, %arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_26,), kwargs = {})
#   %convolution_27 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_26 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_27,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_7 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_26, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_14, %arg28_1, %arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_27 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_28,), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_51 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_51(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/pu/cpurrsc5vq4nye5jwox3u7t6jxacoj2yeqkdwkkzgv3uqttiy6ll.py
# Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66, x_67, x_68, x_69, x_70, x_71], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   sub_1 => sub_1
#   x_36 => div_1
#   x_37 => convolution_16
#   x_38 => relu_15
#   x_39 => convolution_17
#   x_40 => relu_16
#   x_41 => _low_memory_max_pool2d_with_offsets_4
#   x_42 => convolution_18
#   x_43 => relu_17
#   x_44 => convolution_19
#   x_45 => relu_18
#   x_46 => _low_memory_max_pool2d_with_offsets_5
#   x_47 => convolution_20
#   x_48 => relu_19
#   x_49 => convolution_21
#   x_50 => relu_20
#   x_51 => convolution_22
#   x_52 => relu_21
#   x_53 => convolution_23
#   x_54 => relu_22
#   x_55 => _low_memory_max_pool2d_with_offsets_6
#   x_56 => convolution_24
#   x_57 => relu_23
#   x_58 => convolution_25
#   x_59 => relu_24
#   x_60 => convolution_26
#   x_61 => relu_25
#   x_62 => convolution_27
#   x_63 => relu_26
#   x_64 => _low_memory_max_pool2d_with_offsets_7
#   x_65 => convolution_28
#   x_66 => relu_27
#   x_67 => convolution_29
#   x_68 => relu_28
#   x_69 => convolution_30
#   x_70 => relu_29
#   x_71 => convolution_31
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %arg3_1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %arg4_1, %arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg6_1, %arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_18, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg14_1, %arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg16_1, %arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg18_1, %arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %arg20_1, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg22_1, %arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_24 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg24_1, %arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_26,), kwargs = {})
#   %convolution_27 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_26 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_27,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_7 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_26, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_14, %arg28_1, %arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_27 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_28,), kwargs = {})
#   %convolution_29 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_27, %arg30_1, %arg31_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_28 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_29,), kwargs = {})
#   %convolution_30 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_28, %arg32_1, %arg33_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_29 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_30,), kwargs = {})
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_29, %arg34_1, %arg35_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_52 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_52(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 1024*y3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dr/cdrrshtd55lt32vwfihidtwi4x4nt4vc3necjrvl7gyzqbovemvo.py
# Topologically Sorted Source Nodes: [ones], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   ones => full_default_2
# Graph fragment:
#   %full_default_2 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([4, 64, 64], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_ones_53 = async_compile.triton('triton_poi_fused_ones_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_53(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/i7/ci723w6y6bcn5vp5k6uwp5diwjrvkwb3ervxtcpuspptiyyqbl5e.py
# Topologically Sorted Source Nodes: [repeat, mul_1, mul_2], Original ATen: [aten.repeat, aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_2
#   mul_2 => mul_3
#   repeat => repeat
# Graph fragment:
#   %repeat : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%view_1, [4, 1, 1]), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm, %repeat), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm, %repeat), kwargs = {})
triton_poi_fused_mul_repeat_54 = async_compile.triton('triton_poi_fused_mul_repeat_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_repeat_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_repeat_54(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = x1
    tmp2 = x0
    tmp3 = tmp1 == tmp2
    tmp4 = 1.0
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/as/casrn6v36ztvaeh2jdy3pfla4d2rqup6to7wtndj25uspy2nrvoh.py
# Topologically Sorted Source Nodes: [exp, add, mul_3, dcov, dcov_1, dcov_2, add_1, dcov_3], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   dcov => sub_2
#   dcov_1 => clamp_min
#   dcov_2 => mul_5
#   dcov_3 => sqrt
#   exp => full_default_3
#   mul_3 => mul_4
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.000244140625), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_1, %bmm_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm, 2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mul_4), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_3, %clamp_min), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, 1e-05), kwargs = {})
#   %sqrt : [num_users=4] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1,), kwargs = {})
triton_poi_fused_add_clamp_exp_mul_sqrt_sub_55 = async_compile.triton('triton_poi_fused_add_clamp_exp_mul_sqrt_sub_55', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_exp_mul_sqrt_sub_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_exp_mul_sqrt_sub_55(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 0.000244140625
    tmp10 = tmp9 * tmp8
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tl.store(in_out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/6n/c6n2vcodxy72aebth35l3zhk73tn4gqcqbglj6lknyqgua4d3xsv.py
# Topologically Sorted Source Nodes: [mul_5, sub_3, mul_6, sub_4, mul_7, dcdm, mul_13, sub_6, mul_14, sub_7, mul_15, dcdm_1, mul_16, Gamma_XY, mul_17, Gamma_XX, mul_18, Gamma_YY, dc_scores], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum, aten.stack]
# Source node to ATen node mapping:
#   Gamma_XX => sum_2
#   Gamma_XY => sum_1
#   Gamma_YY => sum_3
#   dc_scores => cat
#   dcdm => add_2
#   dcdm_1 => add_5
#   mul_13 => mul_15
#   mul_14 => mul_16
#   mul_15 => mul_17
#   mul_16 => mul_18
#   mul_17 => mul_19
#   mul_18 => mul_20
#   mul_5 => mul_6
#   mul_6 => mul_7
#   mul_7 => mul_8
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_6 => sub_6
#   sub_7 => sub_7
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, 0.015625), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt, %mul_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_4, 0.015625), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %mul_7), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_6, 0.000244140625), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_4, %mul_8), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_10, 0.015625), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_1, %mul_15), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_11, 0.015625), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_6, %mul_16), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_13, 0.000244140625), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_7, %mul_17), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %add_5), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_18, [1, 2]), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %add_2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_19, [1, 2]), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %add_5), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [1, 2]), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_10, %unsqueeze_11, %unsqueeze_12, %unsqueeze_13, %unsqueeze_14], 1), kwargs = {})
triton_red_fused_add_mul_stack_sub_sum_56 = async_compile.triton('triton_red_fused_add_mul_stack_sub_sum_56', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_stack_sub_sum_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_stack_sub_sum_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr5 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr6 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr7 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.015625
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 - tmp3
        tmp6 = tmp5 * tmp2
        tmp7 = tmp4 - tmp6
        tmp9 = 0.000244140625
        tmp10 = tmp8 * tmp9
        tmp11 = tmp7 + tmp10
        tmp14 = tmp13 * tmp2
        tmp15 = tmp12 - tmp14
        tmp17 = tmp16 * tmp2
        tmp18 = tmp15 - tmp17
        tmp20 = tmp19 * tmp9
        tmp21 = tmp18 + tmp20
        tmp22 = tmp11 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp26 = tmp11 * tmp11
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tmp30 = tmp21 * tmp21
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tmp34 = 1e-06
    tmp35 = tmp24 + tmp34
    tmp36 = tmp28 * tmp32
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tmp37 + tmp34
    tmp39 = tmp35 / tmp38
    tl.store(out_ptr3 + (5*x0), tmp39, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zq/czqm4qmckoybde6vhytq3zp4e62222goty27y3cbs2gkx25w63po.py
# Topologically Sorted Source Nodes: [mean, score], Original ATen: [aten.mean, aten.rsub]
# Source node to ATen node mapping:
#   mean => mean
#   score => sub_32
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat, [1], True), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mean), kwargs = {})
triton_poi_fused_mean_rsub_57 = async_compile.triton('triton_poi_fused_mean_rsub_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_rsub_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_rsub_57(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 5*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 5*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 5*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 5*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = 5.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(arg1_1, (4, 3, 512, 512), (786432, 262144, 512, 1))
    assert_size_stride(arg2_1, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(arg3_1, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(arg4_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg35_1, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf106 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones_2], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_0.run(buf106, 65536, grid=grid(65536), stream=stream0)
        buf11 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf51 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_1.run(arg10_1, buf11, buf51, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg10_1
        buf8 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        buf48 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_2.run(arg8_1, buf8, buf48, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg8_1
        buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        buf44 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, sub_1, x_36, x_37, x_38, x_39], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_relu_sub_3.run(arg6_1, buf4, buf44, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg6_1
        buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        buf41 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, sub_1, x_36, x_37], Original ATen: [aten.sub, aten.div, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_sub_4.run(arg4_1, buf1, buf41, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg4_1
        buf0 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x], Original ATen: [aten.sub, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sub_5.run(arg0_1, arg2_1, arg3_1, buf0, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [sub, x, x_1], Original ATen: [aten.sub, aten.div, aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_relu_sub_6.run(buf3, arg5_1, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf4
        buf6 = reinterpret_tensor(buf3, (4, 64, 64, 64), (262144, 4096, 64, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_relu_sub_7.run(buf5, arg7_1, buf6, 256, 4096, grid=grid(256, 4096), stream=stream0)
        buf7 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_8.run(buf6, buf7, 256, 1024, grid=grid(256, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf8
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9.run(buf10, arg9_1, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf11
        buf13 = reinterpret_tensor(buf10, (4, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_10.run(buf12, arg11_1, buf13, 512, 1024, grid=grid(512, 1024), stream=stream0)
        del buf12
        buf105 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_pow2_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (4, 128, 1024), (131072, 1024, 1), 0), reinterpret_tensor(buf13, (4, 1024, 128), (131072, 1, 1024), 0), out=buf105)
        buf107 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        buf109 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [repeat_2, mul_21, mul_22], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_11.run(buf105, buf107, buf109, 65536, grid=grid(65536), stream=stream0)
        buf108 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones_2, repeat_2, mul_21, bmm_15], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf106, buf107, out=buf108)
        buf110 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [repeat_2, mul_22, bmm_16], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf109, buf106, out=buf110)
        buf111 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [exp_2, add_8, mul_23, dcov_8, dcov_9, dcov_10, add_9, dcov_11], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_12.run(buf111, buf110, buf105, 65536, grid=grid(65536), stream=stream0)
        buf112 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [bmm_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf111, buf106, out=buf112)
        buf113 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [bmm_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf106, buf111, out=buf113)
        buf114 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [bmm_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf106, buf111, out=buf114)
        buf115 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf114, buf106, out=buf115)
        buf117 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [ones_3], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_0.run(buf117, 65536, grid=grid(65536), stream=stream0)
        buf40 = empty_strided_cuda((4, 3, 512, 512), (786432, 1, 1536, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, x_36], Original ATen: [aten.sub, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sub_13.run(arg1_1, arg2_1, arg3_1, buf40, 12, 262144, grid=grid(12, 262144), stream=stream0)
        del arg1_1
        del arg2_1
        del arg3_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37], Original ATen: [aten.sub, aten.div, aten.convolution]
        buf42 = extern_kernels.convolution(buf40, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 64, 512, 512), (16777216, 1, 32768, 64))
        del buf40
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_relu_sub_14.run(buf43, arg5_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg5_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 64, 512, 512), (16777216, 1, 32768, 64))
        del buf44
        buf46 = reinterpret_tensor(buf43, (4, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_relu_sub_15.run(buf45, arg7_1, buf46, 256, 262144, grid=grid(256, 262144), stream=stream0)
        del arg7_1
        del buf45
        buf47 = empty_strided_cuda((4, 64, 256, 256), (4194304, 1, 16384, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_16.run(buf46, buf47, 256, 65536, grid=grid(256, 65536), stream=stream0)
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf49 = extern_kernels.convolution(buf47, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 128, 256, 256), (8388608, 1, 32768, 128))
        del buf47
        del buf48
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_17.run(buf50, arg9_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg9_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 128, 256, 256), (8388608, 1, 32768, 128))
        del buf51
        buf53 = reinterpret_tensor(buf50, (4, 128, 256, 256), (8388608, 65536, 256, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_18.run(buf52, arg11_1, buf53, 512, 65536, grid=grid(512, 65536), stream=stream0)
        del arg11_1
        del buf52
        buf116 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_pow2_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (4, 128, 65536), (8388608, 65536, 1), 0), reinterpret_tensor(buf53, (4, 65536, 128), (8388608, 1, 65536), 0), out=buf116)
        buf118 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        buf120 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [repeat_3, mul_29, mul_30], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_11.run(buf116, buf118, buf120, 65536, grid=grid(65536), stream=stream0)
        buf119 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones_3, repeat_3, mul_29, bmm_22], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf117, buf118, out=buf119)
        buf121 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [repeat_3, mul_30, bmm_23], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf120, buf117, out=buf121)
        buf122 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [exp_3, add_11, mul_31, dcov_12, dcov_13, dcov_14, add_12, dcov_15], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_12.run(buf122, buf121, buf116, 65536, grid=grid(65536), stream=stream0)
        buf123 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [bmm_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf122, buf117, out=buf123)
        buf124 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [bmm_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf117, buf122, out=buf124)
        buf125 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [bmm_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf117, buf122, out=buf125)
        buf126 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf125, buf117, out=buf126)
        del buf117
        del buf125
        buf127 = empty_strided_cuda((4, 2), (2, 1), torch.float32)
        buf129 = empty_strided_cuda((4, 2), (2, 1), torch.float32)
        buf131 = empty_strided_cuda((4, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_25, sub_9, mul_26, sub_10, mul_27, dcdm_2, mul_33, sub_12, mul_34, sub_13, mul_35, dcdm_3, mul_36, Gamma_XY_1, mul_37, Gamma_XX_1, mul_38, Gamma_YY_1], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sub_sum_19.run(buf111, buf112, buf113, buf115, buf122, buf123, buf124, buf126, buf127, buf129, buf131, 8, 8192, grid=grid(8), stream=stream0)
        del buf111
        del buf112
        del buf113
        del buf115
        del buf122
        del buf123
        del buf124
        buf248 = empty_strided_cuda((4, 5), (5, 1), torch.float32)
        buf244 = reinterpret_tensor(buf248, (4, 1), (5, 1), 1)  # alias
        # Topologically Sorted Source Nodes: [mul_25, sub_9, mul_26, sub_10, mul_27, dcdm_2, mul_33, sub_12, mul_34, sub_13, mul_35, dcdm_3, mul_36, Gamma_XY_1, mul_37, Gamma_XX_1, mul_38, Gamma_YY_1, dc_scores], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mul_stack_sub_sum_20.run(buf127, buf129, buf131, buf244, 4, 2, grid=grid(4), stream=stream0)
        del buf127
        del buf129
        del buf131
        buf134 = reinterpret_tensor(buf7, (4, 256, 256), (65536, 256, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [ones_4], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_21.run(buf134, 262144, grid=grid(262144), stream=stream0)
        buf15 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf55 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_22.run(arg12_1, buf15, buf55, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg12_1
        buf14 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_23.run(buf13, buf14, 512, 256, grid=grid(512, 256), stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf14
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_24.run(buf17, arg13_1, 262144, grid=grid(262144), stream=stream0)
        buf18 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        buf58 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25.run(arg14_1, buf18, buf58, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg14_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_24.run(buf20, arg15_1, 262144, grid=grid(262144), stream=stream0)
        buf21 = buf18; del buf18  # reuse
        buf61 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25.run(arg16_1, buf21, buf61, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg16_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_24.run(buf23, arg17_1, 262144, grid=grid(262144), stream=stream0)
        buf24 = buf21; del buf21  # reuse
        buf64 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25.run(arg18_1, buf24, buf64, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg18_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf25 = extern_kernels.convolution(buf23, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf24
        buf26 = reinterpret_tensor(buf23, (4, 256, 16, 16), (65536, 256, 16, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_26.run(buf25, arg19_1, buf26, 1024, 256, grid=grid(1024, 256), stream=stream0)
        buf133 = reinterpret_tensor(buf25, (4, 256, 256), (65536, 256, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_pow2_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (4, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf26, (4, 256, 256), (65536, 1, 256), 0), out=buf133)
        buf135 = reinterpret_tensor(buf20, (4, 256, 256), (65536, 256, 1), 0); del buf20  # reuse
        buf137 = reinterpret_tensor(buf17, (4, 256, 256), (65536, 256, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [repeat_4, mul_41, mul_42], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_27.run(buf133, buf135, buf137, 262144, grid=grid(262144), stream=stream0)
        buf136 = empty_strided_cuda((4, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones_4, repeat_4, mul_41, bmm_29], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf134, buf135, out=buf136)
        buf138 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [repeat_4, mul_42, bmm_30], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf137, buf134, out=buf138)
        buf139 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [exp_4, add_16, mul_43, dcov_16, dcov_17, dcov_18, add_17, dcov_19], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_28.run(buf139, buf138, buf133, 262144, grid=grid(262144), stream=stream0)
        buf140 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [bmm_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf139, buf134, out=buf140)
        buf141 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [bmm_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf134, buf139, out=buf141)
        buf142 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [bmm_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf134, buf139, out=buf142)
        buf143 = empty_strided_cuda((4, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf142, buf134, out=buf143)
        buf145 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [ones_5], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_21.run(buf145, 262144, grid=grid(262144), stream=stream0)
        buf54 = empty_strided_cuda((4, 128, 128, 128), (2097152, 1, 16384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_29.run(buf53, buf54, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del buf53
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf56 = extern_kernels.convolution(buf54, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf54
        del buf55
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_30.run(buf57, arg13_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg13_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf59 = extern_kernels.convolution(buf57, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf57
        del buf58
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_30.run(buf60, arg15_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg15_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf60
        del buf61
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_30.run(buf63, arg17_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg17_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf65 = extern_kernels.convolution(buf63, buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        del buf64
        buf66 = reinterpret_tensor(buf63, (4, 256, 128, 128), (4194304, 16384, 128, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31.run(buf65, arg19_1, buf66, 1024, 16384, grid=grid(1024, 16384), stream=stream0)
        del arg19_1
        del buf65
        buf144 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_pow2_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf66, (4, 256, 16384), (4194304, 16384, 1), 0), reinterpret_tensor(buf66, (4, 16384, 256), (4194304, 1, 16384), 0), out=buf144)
        buf146 = empty_strided_cuda((4, 256, 256), (65536, 256, 1), torch.float32)
        buf148 = empty_strided_cuda((4, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [repeat_5, mul_49, mul_50], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_27.run(buf144, buf146, buf148, 262144, grid=grid(262144), stream=stream0)
        buf147 = empty_strided_cuda((4, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones_5, repeat_5, mul_49, bmm_36], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf145, buf146, out=buf147)
        buf149 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [repeat_5, mul_50, bmm_37], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf148, buf145, out=buf149)
        buf150 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [exp_5, add_19, mul_51, dcov_20, dcov_21, dcov_22, add_20, dcov_23], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_28.run(buf150, buf149, buf144, 262144, grid=grid(262144), stream=stream0)
        buf151 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [bmm_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf150, buf145, out=buf151)
        buf152 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [bmm_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf145, buf150, out=buf152)
        buf153 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [bmm_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf145, buf150, out=buf153)
        buf154 = empty_strided_cuda((4, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf153, buf145, out=buf154)
        del buf145
        del buf153
        buf155 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        buf157 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        buf159 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_45, sub_15, mul_46, sub_16, mul_47, dcdm_4, mul_53, sub_18, mul_54, sub_19, mul_55, dcdm_5, mul_56, Gamma_XY_2, mul_57, Gamma_XX_2, mul_58, Gamma_YY_2], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sub_sum_32.run(buf139, buf140, buf141, buf143, buf150, buf151, buf152, buf154, buf155, buf157, buf159, 32, 8192, grid=grid(32), stream=stream0)
        del buf139
        del buf140
        del buf141
        del buf143
        del buf150
        del buf151
        del buf152
        del buf154
        buf245 = reinterpret_tensor(buf248, (4, 1), (5, 1), 2)  # alias
        # Topologically Sorted Source Nodes: [mul_45, sub_15, mul_46, sub_16, mul_47, dcdm_4, mul_53, sub_18, mul_54, sub_19, mul_55, dcdm_5, mul_56, Gamma_XY_2, mul_57, Gamma_XX_2, mul_58, Gamma_YY_2, dc_scores], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mul_stack_sub_sum_33.run(buf155, buf157, buf159, buf245, 4, 8, grid=grid(4), stream=stream0)
        del buf155
        del buf157
        del buf159
        buf162 = reinterpret_tensor(buf5, (4, 512, 512), (262144, 512, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [ones_6], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_34.run(buf162, 1048576, grid=grid(1048576), stream=stream0)
        buf27 = reinterpret_tensor(buf126, (4, 256, 8, 8), (16384, 1, 2048, 256), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_35.run(buf26, buf27, 1024, 64, grid=grid(1024, 64), stream=stream0)
        del buf26
        buf28 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        buf68 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_36.run(arg20_1, buf28, buf68, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg20_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf27
        del buf28
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_37.run(buf30, arg21_1, 131072, grid=grid(131072), stream=stream0)
        buf31 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        buf71 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38.run(arg22_1, buf31, buf71, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg22_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf32 = extern_kernels.convolution(buf30, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf30
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_37.run(buf33, arg23_1, 131072, grid=grid(131072), stream=stream0)
        buf34 = buf31; del buf31  # reuse
        buf74 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38.run(arg24_1, buf34, buf74, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg24_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf35 = extern_kernels.convolution(buf33, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf33
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_37.run(buf36, arg25_1, 131072, grid=grid(131072), stream=stream0)
        buf37 = buf34; del buf34  # reuse
        buf77 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38.run(arg26_1, buf37, buf77, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg26_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf39 = reinterpret_tensor(buf36, (4, 512, 8, 8), (32768, 64, 8, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_39.run(buf38, arg27_1, buf39, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del buf38
        buf161 = empty_strided_cuda((4, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_pow2_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf39, (4, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf39, (4, 64, 512), (32768, 1, 64), 0), out=buf161)
        buf163 = empty_strided_cuda((4, 512, 512), (262144, 512, 1), torch.float32)
        buf165 = empty_strided_cuda((4, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [repeat_6, mul_61, mul_62], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_40.run(buf161, buf163, buf165, 1048576, grid=grid(1048576), stream=stream0)
        buf164 = empty_strided_cuda((4, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones_6, repeat_6, mul_61, bmm_43], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf162, buf163, out=buf164)
        buf166 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [repeat_6, mul_62, bmm_44], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf165, buf162, out=buf166)
        buf167 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [exp_6, add_24, mul_63, dcov_24, dcov_25, dcov_26, add_25, dcov_27], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_41.run(buf167, buf166, buf161, 1048576, grid=grid(1048576), stream=stream0)
        buf168 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [bmm_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf167, buf162, out=buf168)
        buf169 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [bmm_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf162, buf167, out=buf169)
        buf170 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [bmm_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf162, buf167, out=buf170)
        buf171 = empty_strided_cuda((4, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf170, buf162, out=buf171)
        buf173 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [ones_7], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_34.run(buf173, 1048576, grid=grid(1048576), stream=stream0)
        buf67 = empty_strided_cuda((4, 256, 64, 64), (1048576, 1, 16384, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_42.run(buf66, buf67, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del buf66
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf69 = extern_kernels.convolution(buf67, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        del buf67
        del buf68
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_43.run(buf70, arg21_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf72 = extern_kernels.convolution(buf70, buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        del buf70
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_43.run(buf73, arg23_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg23_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf75 = extern_kernels.convolution(buf73, buf74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        del buf73
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_43.run(buf76, arg25_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg25_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf78 = extern_kernels.convolution(buf76, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        buf79 = reinterpret_tensor(buf76, (4, 512, 64, 64), (2097152, 4096, 64, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_44.run(buf78, arg27_1, buf79, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        del arg27_1
        del buf78
        buf172 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_pow2_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (4, 512, 4096), (2097152, 4096, 1), 0), reinterpret_tensor(buf79, (4, 4096, 512), (2097152, 1, 4096), 0), out=buf172)
        buf174 = empty_strided_cuda((4, 512, 512), (262144, 512, 1), torch.float32)
        buf176 = empty_strided_cuda((4, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [repeat_7, mul_69, mul_70], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_40.run(buf172, buf174, buf176, 1048576, grid=grid(1048576), stream=stream0)
        buf175 = empty_strided_cuda((4, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones_7, repeat_7, mul_69, bmm_50], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf173, buf174, out=buf175)
        buf177 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [repeat_7, mul_70, bmm_51], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf176, buf173, out=buf177)
        buf178 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [exp_7, add_27, mul_71, dcov_28, dcov_29, dcov_30, add_28, dcov_31], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_41.run(buf178, buf177, buf172, 1048576, grid=grid(1048576), stream=stream0)
        buf179 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [bmm_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf178, buf173, out=buf179)
        buf180 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [bmm_53], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf173, buf178, out=buf180)
        buf181 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [bmm_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf173, buf178, out=buf181)
        buf182 = empty_strided_cuda((4, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf181, buf173, out=buf182)
        buf183 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        buf185 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        buf187 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_65, sub_21, mul_66, sub_22, mul_67, dcdm_6, mul_73, sub_24, mul_74, sub_25, mul_75, dcdm_7, mul_76, Gamma_XY_3, mul_77, Gamma_XX_3, mul_78, Gamma_YY_3], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sub_sum_45.run(buf167, buf168, buf169, buf171, buf178, buf179, buf180, buf182, buf183, buf185, buf187, 128, 8192, grid=grid(128), stream=stream0)
        buf246 = reinterpret_tensor(buf248, (4, 1), (5, 1), 3)  # alias
        # Topologically Sorted Source Nodes: [mul_65, sub_21, mul_66, sub_22, mul_67, dcdm_6, mul_73, sub_24, mul_74, sub_25, mul_75, dcdm_7, mul_76, Gamma_XY_3, mul_77, Gamma_XX_3, mul_78, Gamma_YY_3, dc_scores], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mul_stack_sub_sum_46.run(buf183, buf185, buf187, buf246, 4, 32, grid=grid(4), stream=stream0)
        buf203 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [ones_8], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_34.run(buf203, 1048576, grid=grid(1048576), stream=stream0)
        buf189 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_47.run(buf39, buf189, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del buf39
        buf190 = buf77; del buf77  # reuse
        buf214 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_29, x_65], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38.run(arg28_1, buf190, buf214, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf191 = extern_kernels.convolution(buf189, buf190, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf189
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_48.run(buf192, arg29_1, 32768, grid=grid(32768), stream=stream0)
        buf193 = buf190; del buf190  # reuse
        buf217 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_29, x_30, x_31, x_65, x_66, x_67], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38.run(arg30_1, buf193, buf217, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg30_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf194 = extern_kernels.convolution(buf192, buf193, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf192
        buf195 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_48.run(buf195, arg31_1, 32768, grid=grid(32768), stream=stream0)
        buf196 = buf193; del buf193  # reuse
        buf220 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_29, x_30, x_31, x_32, x_33, x_65, x_66, x_67, x_68, x_69], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38.run(arg32_1, buf196, buf220, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg32_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf197 = extern_kernels.convolution(buf195, buf196, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf195
        buf198 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33, x_34], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_48.run(buf198, arg33_1, 32768, grid=grid(32768), stream=stream0)
        buf199 = buf196; del buf196  # reuse
        buf223 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_29, x_30, x_31, x_32, x_33, x_34, x_35, x_65, x_66, x_67, x_68, x_69, x_70, x_71], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_38.run(arg34_1, buf199, buf223, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg34_1
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33, x_34, x_35], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf200 = extern_kernels.convolution(buf198, buf199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf199
        buf201 = reinterpret_tensor(buf198, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [sub, x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33, x_34, x_35], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_49.run(buf200, arg35_1, buf201, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del buf200
        buf202 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_pow2_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (4, 512, 16), (8192, 16, 1), 0), reinterpret_tensor(buf201, (4, 16, 512), (8192, 1, 16), 0), out=buf202)
        del buf201
        buf204 = buf179; del buf179  # reuse
        buf206 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [repeat_8, mul_81, mul_82], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_40.run(buf202, buf204, buf206, 1048576, grid=grid(1048576), stream=stream0)
        buf205 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [ones_8, repeat_8, mul_81, bmm_57], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf203, buf204, out=buf205)
        buf207 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [repeat_8, mul_82, bmm_58], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf206, buf203, out=buf207)
        buf208 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [exp_8, add_32, mul_83, dcov_32, dcov_33, dcov_34, add_33, dcov_35], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_41.run(buf208, buf207, buf202, 1048576, grid=grid(1048576), stream=stream0)
        buf209 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [bmm_59], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf208, buf203, out=buf209)
        buf210 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [bmm_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf203, buf208, out=buf210)
        buf211 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [bmm_61], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf203, buf208, out=buf211)
        buf212 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [bmm_62], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf211, buf203, out=buf212)
        buf227 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [ones_9], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_34.run(buf227, 1048576, grid=grid(1048576), stream=stream0)
        buf213 = empty_strided_cuda((4, 512, 32, 32), (524288, 1, 16384, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_50.run(buf79, buf213, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        del buf79
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf215 = extern_kernels.convolution(buf213, buf214, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf213
        del buf214
        buf216 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_51.run(buf216, arg29_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66, x_67], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf218 = extern_kernels.convolution(buf216, buf217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf216
        del buf217
        buf219 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66, x_67, x_68], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_51.run(buf219, arg31_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg31_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66, x_67, x_68, x_69], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf221 = extern_kernels.convolution(buf219, buf220, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf219
        del buf220
        buf222 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66, x_67, x_68, x_69, x_70], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_51.run(buf222, arg33_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg33_1
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66, x_67, x_68, x_69, x_70, x_71], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf224 = extern_kernels.convolution(buf222, buf223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 512, 32, 32), (524288, 1, 16384, 512))
        del buf223
        buf225 = reinterpret_tensor(buf222, (4, 512, 32, 32), (524288, 1024, 32, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66, x_67, x_68, x_69, x_70, x_71], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_52.run(buf224, arg35_1, buf225, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        del arg35_1
        del buf224
        buf226 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [x_pow2_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (4, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(buf225, (4, 1024, 512), (524288, 1, 1024), 0), out=buf226)
        del buf225
        buf228 = buf168; del buf168  # reuse
        buf230 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [repeat_9, mul_89, mul_90], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_40.run(buf226, buf228, buf230, 1048576, grid=grid(1048576), stream=stream0)
        buf229 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [ones_9, repeat_9, mul_89, bmm_64], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf227, buf228, out=buf229)
        buf231 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [repeat_9, mul_90, bmm_65], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf230, buf227, out=buf231)
        buf232 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [exp_9, add_35, mul_91, dcov_36, dcov_37, dcov_38, add_36, dcov_39], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_41.run(buf232, buf231, buf226, 1048576, grid=grid(1048576), stream=stream0)
        buf233 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [bmm_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf232, buf227, out=buf233)
        buf234 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [bmm_67], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf227, buf232, out=buf234)
        buf235 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [bmm_68], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf227, buf232, out=buf235)
        buf236 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [bmm_69], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf235, buf227, out=buf236)
        del buf227
        del buf235
        buf237 = buf187; del buf187  # reuse
        buf239 = buf185; del buf185  # reuse
        buf241 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [mul_85, sub_27, mul_86, sub_28, mul_87, dcdm_8, mul_93, sub_30, mul_94, sub_31, mul_95, dcdm_9, mul_96, Gamma_XY_4, mul_97, Gamma_XX_4, mul_98, Gamma_YY_4], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sub_sum_45.run(buf208, buf209, buf210, buf212, buf232, buf233, buf234, buf236, buf237, buf239, buf241, 128, 8192, grid=grid(128), stream=stream0)
        del buf208
        del buf209
        del buf210
        del buf212
        del buf232
        del buf233
        del buf234
        del buf236
        buf247 = reinterpret_tensor(buf248, (4, 1), (5, 1), 4)  # alias
        # Topologically Sorted Source Nodes: [mul_85, sub_27, mul_86, sub_28, mul_87, dcdm_8, mul_93, sub_30, mul_94, sub_31, mul_95, dcdm_9, mul_96, Gamma_XY_4, mul_97, Gamma_XX_4, mul_98, Gamma_YY_4, dc_scores], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mul_stack_sub_sum_46.run(buf237, buf239, buf241, buf247, 4, 32, grid=grid(4), stream=stream0)
        del buf237
        del buf239
        del buf241
        buf81 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_53.run(buf81, 16384, grid=grid(16384), stream=stream0)
        buf80 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_pow2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (4, 64, 4096), (262144, 4096, 1), 0), reinterpret_tensor(buf6, (4, 4096, 64), (262144, 1, 4096), 0), out=buf80)
        del buf6
        buf82 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        buf84 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [repeat, mul_1, mul_2], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_54.run(buf80, buf82, buf84, 16384, grid=grid(16384), stream=stream0)
        buf83 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones, repeat, mul_1, bmm_1], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf81, buf82, out=buf83)
        buf85 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [repeat, mul_2, bmm_2], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf84, buf81, out=buf85)
        buf86 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [exp, add, mul_3, dcov, dcov_1, dcov_2, add_1, dcov_3], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_55.run(buf86, buf85, buf80, 16384, grid=grid(16384), stream=stream0)
        buf87 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [bmm_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf86, buf81, out=buf87)
        buf88 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [bmm_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf81, buf86, out=buf88)
        buf89 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [bmm_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf81, buf86, out=buf89)
        buf90 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf89, buf81, out=buf90)
        buf92 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [ones_1], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_53.run(buf92, 16384, grid=grid(16384), stream=stream0)
        buf91 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_pow2_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (4, 64, 262144), (16777216, 262144, 1), 0), reinterpret_tensor(buf46, (4, 262144, 64), (16777216, 1, 262144), 0), out=buf91)
        del buf46
        buf93 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        buf95 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [repeat_1, mul_9, mul_10], Original ATen: [aten.repeat, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_repeat_54.run(buf91, buf93, buf95, 16384, grid=grid(16384), stream=stream0)
        buf94 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones_1, repeat_1, mul_9, bmm_8], Original ATen: [aten.ones, aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf92, buf93, out=buf94)
        buf96 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [repeat_1, mul_10, bmm_9], Original ATen: [aten.repeat, aten.mul, aten.bmm]
        extern_kernels.bmm(buf95, buf92, out=buf96)
        buf97 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [exp_1, add_3, mul_11, dcov_4, dcov_5, dcov_6, add_4, dcov_7], Original ATen: [aten.exp, aten.add, aten.mul, aten.sub, aten.clamp, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_sqrt_sub_55.run(buf97, buf96, buf91, 16384, grid=grid(16384), stream=stream0)
        buf98 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [bmm_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf97, buf92, out=buf98)
        buf99 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [bmm_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf92, buf97, out=buf99)
        buf100 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [bmm_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf92, buf97, out=buf100)
        buf101 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf100, buf92, out=buf101)
        del buf100
        del buf92
        buf243 = reinterpret_tensor(buf248, (4, 1), (5, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [mul_5, sub_3, mul_6, sub_4, mul_7, dcdm, mul_13, sub_6, mul_14, sub_7, mul_15, dcdm_1, mul_16, Gamma_XY, mul_17, Gamma_XX, mul_18, Gamma_YY, dc_scores], Original ATen: [aten.mul, aten.sub, aten.add, aten.sum, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_stack_sub_sum_56.run(buf86, buf87, buf88, buf90, buf97, buf98, buf99, buf101, buf243, 4, 4096, grid=grid(4), stream=stream0)
        del buf101
        del buf86
        del buf87
        del buf88
        del buf90
        del buf97
        del buf98
        del buf99
        buf249 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mean, score], Original ATen: [aten.mean, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_rsub_57.run(buf248, buf249, 4, grid=grid(4), stream=stream0)
        del buf243
        del buf244
        del buf245
        del buf246
        del buf247
        del buf248
    return (buf249, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 3, 512, 512), (786432, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
