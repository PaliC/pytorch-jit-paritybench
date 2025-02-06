# AOT ID: ['145_inference']
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


# kernel path: inductor_cache/bp/cbpnv2tnckulltjkau4h2jv4qw222h5pc5ihn5oaba5wzel5z2mk.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ho/chofbsued7wdrygtn7zdydvbngfpahgonng27rq3t34k2xgxiqc4.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: inductor_cache/st/cstaarsnbemu4daenwrw2iv3lbe5sphuamdxrw5evpkjhyzh5pvz.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 961
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 61504*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 961*y3), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/m4/cm47di5b2w2qwg4vl4uzhuj4wbisqtwo3uicqw7zfj7bdjfhcobp.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_3 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu, [3, 3], [2, 2], [0, 0], [1, 1], True), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_3 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 15)
    x3 = xindex // 15
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (2*x2 + 62*x3 + 961*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 62*x3 + 961*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x2 + 62*x3 + 961*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (31 + 2*x2 + 62*x3 + 961*y4), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (32 + 2*x2 + 62*x3 + 961*y4), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (33 + 2*x2 + 62*x3 + 961*y4), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (62 + 2*x2 + 62*x3 + 961*y4), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (63 + 2*x2 + 62*x3 + 961*y4), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (64 + 2*x2 + 62*x3 + 961*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + 64*x5 + 14400*y1), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/vu/cvub6gj4fb2ogtno7ecxtc54lt6gwlfhiyypxo4tk7ijjrgnmaio.py
# Topologically Sorted Source Nodes: [conv2d_1, x], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
#   x => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
triton_poi_fused_convolution_relu_4 = async_compile.triton('triton_poi_fused_convolution_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7k/c7kipvnmhlbs6vgv5qzocxhneh5xo3xdfhge2onac4xt2zs6if6q.py
# Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_3 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 144*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ch/cchz6r37eyy6ctrqwv5nd63ore4jgwmjkp3b67ycfy5ltst6cidv.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_4 => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_2, %relu_3], 1), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 115200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 128, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (64*x1 + ((-64) + x0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-64) + x0), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e2/ce2rwr5xfzzfw2jun5dapuyvq2k6zdm63ulklcfsu7l6hsvizwcn.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_5 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_5, %relu_6], 1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 115200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 225) % 128)
    x0 = (xindex % 225)
    x2 = xindex // 28800
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x0 + 14400*x2 + (x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 128, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (64*x0 + 14400*x2 + ((-64) + x1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-64) + x1), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f6/cf6gmvn2wt2ewljjjxqlksi66qqqn3b6yptitdlv2bpdjzz5ovtr.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_6 => _low_memory_max_pool2d_with_offsets_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_8 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 7)
    x3 = xindex // 7
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (2*x2 + 30*x3 + 225*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 30*x3 + 225*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x2 + 30*x3 + 225*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (15 + 2*x2 + 30*x3 + 225*y4), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (16 + 2*x2 + 30*x3 + 225*y4), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (17 + 2*x2 + 30*x3 + 225*y4), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (30 + 2*x2 + 30*x3 + 225*y4), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (31 + 2*x2 + 30*x3 + 225*y4), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (32 + 2*x2 + 30*x3 + 225*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + 128*x5 + 6272*y1), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/bo/cbovstsa5fuef7narmz4sblz77tsqi6gudnt7mhcsjv6feeveo6e.py
# Topologically Sorted Source Nodes: [conv2d_7, x_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_7 => convolution_7
#   x_2 => relu_7
# Graph fragment:
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg15_1, %arg16_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
triton_poi_fused_convolution_relu_9 = async_compile.triton('triton_poi_fused_convolution_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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


# kernel path: inductor_cache/tf/ctf2drlh3uvlioukqsh3xdxhjsbtyi7x2yhbx3632cvo5nx42gov.py
# Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_9 => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_10 = async_compile.triton('triton_poi_fused_convolution_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kj/ckjlgpjsrtjdlbp7rwc6ck6a3hxahydtuc2lioob2w7t7n3ru5eb.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_7 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_8, %relu_9], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 256, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (128*x1 + ((-128) + x0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-128) + x0), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/52/c523s56fau73mabc6ur5rfwrsidgvgw575fqer2nr6giovw3lvpo.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_8 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_11, %relu_12], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 49) % 256)
    x0 = (xindex % 49)
    x2 = xindex // 12544
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x0 + 6272*x2 + (x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 256, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (128*x0 + 6272*x2 + ((-128) + x1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-128) + x1), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ga/cgae3lixf7wjk2r6szhxgpzoqdfduvv3ttcfoiid3xrhpy4xt2y5.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_9 => _low_memory_max_pool2d_with_offsets_2
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_13 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 3)
    x3 = xindex // 3
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (2*x2 + 14*x3 + 49*y4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 14*x3 + 49*y4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x2 + 14*x3 + 49*y4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (7 + 2*x2 + 14*x3 + 49*y4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8 + 2*x2 + 14*x3 + 49*y4), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (9 + 2*x2 + 14*x3 + 49*y4), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (14 + 2*x2 + 14*x3 + 49*y4), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (15 + 2*x2 + 14*x3 + 49*y4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (16 + 2*x2 + 14*x3 + 49*y4), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + 256*x5 + 2304*y1), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/td/ctdjzt5563cnorb3ul76twe7cxnirrbdjotxvk3qabcfr7zurjhb.py
# Topologically Sorted Source Nodes: [conv2d_13, x_4], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_13 => convolution_13
#   x_4 => relu_13
# Graph fragment:
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg27_1, %arg28_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
triton_poi_fused_convolution_relu_14 = async_compile.triton('triton_poi_fused_convolution_relu_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 48)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wa/cwaiqenuskvsx472uzrw6bxvwflad3qrdw7fxvw6h24eshunl3du.py
# Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_15 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg31_1, %arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 48)
    y1 = yindex // 48
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 48*x2 + 432*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2v/c2vfc3kljk3736c5u5dlx76vyjbluulqir4zqfkphhe5euaeofej.py
# Topologically Sorted Source Nodes: [input_10, conv2d_16], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_16 => convolution_16
#   input_10 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_14, %relu_15], 1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_4, %arg33_1, %arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_16 = async_compile.triton('triton_poi_fused_cat_convolution_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 384)
    x2 = xindex
    y1 = yindex // 384
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (192*x2 + 1728*y1 + (y0)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1, 1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 384, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (192*x2 + 1728*y1 + ((-192) + y0)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x2 + 9*y3), tmp22, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 384*x2 + 3456*y1), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ep/cep7uis5xgeswi7xikilkaycsabtkmay5etiv67vh5s2dpzijdc6.py
# Topologically Sorted Source Nodes: [conv2d_19, x_6], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_19 => convolution_19
#   x_6 => relu_19
# Graph fragment:
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_5, %arg39_1, %arg40_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
triton_poi_fused_convolution_relu_17 = async_compile.triton('triton_poi_fused_convolution_relu_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vr/cvrys3b66meo2o3ljwqqnttdwrzoxicqc3diozcu6bj62zsaxpl3.py
# Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_21 => convolution_21
# Graph fragment:
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg43_1, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_18 = async_compile.triton('triton_poi_fused_convolution_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dm/cdmfdvlg2iv3a4r2haan3d4we6vxu6nyqnyrmyabcctwwytb6yhh.py
# Topologically Sorted Source Nodes: [input_12, conv2d_22], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_22 => convolution_22
#   input_12 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_20, %relu_21], 1), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_6, %arg45_1, %arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_19 = async_compile.triton('triton_poi_fused_cat_convolution_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 512)
    x2 = xindex
    y1 = yindex // 512
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x2 + 2304*y1 + (y0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1, 1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 512, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (256*x2 + 2304*y1 + ((-256) + y0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to((-256) + y0, [XBLOCK, YBLOCK])), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x2 + 9*y3), tmp22, xmask)
    tl.store(out_ptr1 + (y0 + 512*x2 + 4608*y1), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oa/coaya4jd2v3rjjo233b4zageaatudhfykjzgao22k5qznomqkkz6.py
# Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_13 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_23, %relu_24], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 9) % 512)
    x0 = (xindex % 9)
    x2 = xindex // 4608
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x0 + 2304*x2 + (x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 512, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (256*x0 + 2304*x2 + ((-256) + x1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-256) + x1), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(arg3_1, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (16, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg10_1, (16, ), (1, ))
    assert_size_stride(arg11_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg16_1, (32, ), (1, ))
    assert_size_stride(arg17_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg22_1, (32, ), (1, ))
    assert_size_stride(arg23_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg28_1, (48, ), (1, ))
    assert_size_stride(arg29_1, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (48, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg34_1, (48, ), (1, ))
    assert_size_stride(arg35_1, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg36_1, (192, ), (1, ))
    assert_size_stride(arg37_1, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg40_1, (64, ), (1, ))
    assert_size_stride(arg41_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg46_1, (64, ), (1, ))
    assert_size_stride(arg47_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg50_1, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg2_1, buf0, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del arg2_1
        buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 31, 31), (61504, 1, 1984, 64))
        del buf0
        del buf1
        buf3 = empty_strided_cuda((4, 64, 31, 31), (61504, 961, 31, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf2, arg1_1, buf3, 256, 961, grid=grid(256, 961), stream=stream0)
        del arg1_1
        del buf2
        buf4 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_3.run(buf3, buf4, 256, 225, grid=grid(256, 225), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 16, 15, 15), (3600, 1, 240, 16))
        del arg3_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1, x], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_4.run(buf6, arg4_1, 14400, grid=grid(14400), stream=stream0)
        del arg4_1
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 64, 15, 15), (14400, 1, 960, 64))
        del arg5_1
        buf8 = empty_strided_cuda((64, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_5.run(arg7_1, buf8, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg7_1
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf6, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 64, 15, 15), (14400, 1, 960, 64))
        del buf6
        buf10 = empty_strided_cuda((4, 128, 15, 15), (28800, 1, 1920, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf7, arg6_1, buf9, arg8_1, buf10, 115200, grid=grid(115200), stream=stream0)
        del arg6_1
        del arg8_1
        del buf7
        del buf9
        # Topologically Sorted Source Nodes: [input_4, conv2d_4], Original ATen: [aten.cat, aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 16, 15, 15), (3600, 1, 240, 16))
        del arg9_1
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_4, conv2d_4, x_1], Original ATen: [aten.cat, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_4.run(buf12, arg10_1, 14400, grid=grid(14400), stream=stream0)
        del arg10_1
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 64, 15, 15), (14400, 1, 960, 64))
        del arg11_1
        buf14 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_5.run(arg13_1, buf14, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg13_1
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf12, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 64, 15, 15), (14400, 1, 960, 64))
        del buf12
        buf16 = reinterpret_tensor(buf10, (4, 128, 15, 15), (28800, 225, 15, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf13, arg12_1, buf15, arg14_1, buf16, 115200, grid=grid(115200), stream=stream0)
        del arg12_1
        del arg14_1
        del buf13
        del buf15
        buf17 = empty_strided_cuda((4, 128, 7, 7), (6272, 1, 896, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_8.run(buf16, buf17, 512, 49, grid=grid(512, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 7, 7), (1568, 1, 224, 32))
        del arg15_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [conv2d_7, x_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_9.run(buf19, arg16_1, 6272, grid=grid(6272), stream=stream0)
        del arg16_1
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 128, 7, 7), (6272, 1, 896, 128))
        del arg17_1
        buf21 = empty_strided_cuda((128, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_10.run(arg19_1, buf21, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg19_1
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf19, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 128, 7, 7), (6272, 1, 896, 128))
        del buf19
        buf23 = empty_strided_cuda((4, 256, 7, 7), (12544, 1, 1792, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf20, arg18_1, buf22, arg20_1, buf23, 50176, grid=grid(50176), stream=stream0)
        del arg18_1
        del arg20_1
        del buf20
        del buf22
        # Topologically Sorted Source Nodes: [input_7, conv2d_10], Original ATen: [aten.cat, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 32, 7, 7), (1568, 1, 224, 32))
        del arg21_1
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [input_7, conv2d_10, x_3], Original ATen: [aten.cat, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_9.run(buf25, arg22_1, 6272, grid=grid(6272), stream=stream0)
        del arg22_1
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg23_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 128, 7, 7), (6272, 1, 896, 128))
        del arg23_1
        buf27 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_10.run(arg25_1, buf27, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg25_1
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf25, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 128, 7, 7), (6272, 1, 896, 128))
        del buf25
        del buf27
        buf29 = reinterpret_tensor(buf23, (4, 256, 7, 7), (12544, 49, 7, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf26, arg24_1, buf28, arg26_1, buf29, 50176, grid=grid(50176), stream=stream0)
        del arg24_1
        del arg26_1
        del buf26
        del buf28
        buf30 = reinterpret_tensor(buf14, (4, 256, 3, 3), (2304, 1, 768, 256), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_13.run(buf29, buf30, 1024, 9, grid=grid(1024, 9), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 48, 3, 3), (432, 1, 144, 48))
        del arg27_1
        del buf30
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [conv2d_13, x_4], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf32, arg28_1, 1728, grid=grid(1728), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg29_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 192, 3, 3), (1728, 1, 576, 192))
        del arg29_1
        buf34 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_15.run(arg31_1, buf34, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg31_1
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf32, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 192, 3, 3), (1728, 1, 576, 192))
        del buf32
        buf36 = empty_strided_cuda((4, 384, 3, 3), (3456, 9, 3, 1), torch.float32)
        buf37 = empty_strided_cuda((4, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, conv2d_16], Original ATen: [aten.cat, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_16.run(buf33, arg30_1, buf35, arg32_1, buf36, buf37, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del arg30_1
        del arg32_1
        del buf33
        del buf35
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 48, 3, 3), (432, 1, 144, 48))
        del arg33_1
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [conv2d_16, x_5], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf39, arg34_1, 1728, grid=grid(1728), stream=stream0)
        del arg34_1
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg35_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 192, 3, 3), (1728, 1, 576, 192))
        del arg35_1
        buf41 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_15.run(arg37_1, buf41, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg37_1
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf39, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 192, 3, 3), (1728, 1, 576, 192))
        del buf39
        del buf41
        buf43 = reinterpret_tensor(buf37, (4, 384, 3, 3), (3456, 9, 3, 1), 0); del buf37  # reuse
        buf44 = empty_strided_cuda((4, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, conv2d_19], Original ATen: [aten.cat, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_16.run(buf40, arg36_1, buf42, arg38_1, buf43, buf44, 1536, 9, grid=grid(1536, 9), stream=stream0)
        del arg36_1
        del arg38_1
        del buf40
        del buf42
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 64, 3, 3), (576, 1, 192, 64))
        del arg39_1
        del buf44
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [conv2d_19, x_6], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_17.run(buf46, arg40_1, 2304, grid=grid(2304), stream=stream0)
        del arg40_1
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 256, 3, 3), (2304, 1, 768, 256))
        del arg41_1
        buf48 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_18.run(arg43_1, buf48, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg43_1
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf46, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 256, 3, 3), (2304, 1, 768, 256))
        del buf46
        buf50 = empty_strided_cuda((4, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        buf51 = empty_strided_cuda((4, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_12, conv2d_22], Original ATen: [aten.cat, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_19.run(buf47, arg42_1, buf49, arg44_1, buf50, buf51, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg42_1
        del arg44_1
        del buf47
        del buf49
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 3, 3), (576, 1, 192, 64))
        del arg45_1
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [conv2d_22, x_7], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_17.run(buf53, arg46_1, 2304, grid=grid(2304), stream=stream0)
        del arg46_1
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg47_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 256, 3, 3), (2304, 1, 768, 256))
        del arg47_1
        buf55 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_18.run(arg49_1, buf55, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg49_1
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf53, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 256, 3, 3), (2304, 1, 768, 256))
        del buf53
        del buf55
        buf57 = reinterpret_tensor(buf51, (4, 512, 3, 3), (4608, 9, 3, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf54, arg48_1, buf56, arg50_1, buf57, 18432, grid=grid(18432), stream=stream0)
        del arg48_1
        del arg50_1
        del buf54
        del buf56
    return (buf3, buf16, buf29, buf36, buf43, buf50, buf57, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((48, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
