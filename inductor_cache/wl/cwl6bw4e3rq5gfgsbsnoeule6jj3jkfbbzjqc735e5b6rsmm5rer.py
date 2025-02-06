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


# kernel path: inductor_cache/3k/c3kpmpzpjgk6fzqfteovhhxl7rygakalvm5sdlqokbzxby2bibjm.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   x => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad2d_0 = async_compile.triton('triton_poi_fused_reflection_pad2d_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 58800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 70)
    x1 = ((xindex // 70) % 70)
    x2 = xindex // 4900
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-3) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-3) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i3/ci3pkhutzsjztg7dcyyikhnaffkokhxrce7fnkbmpotjf6rklyfj.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   x_2 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %le_316 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_1 = async_compile.triton('triton_poi_fused_relu_threshold_backward_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/2g/c2g4r3g5d2wovczoefdmmn4th2ximunpuukdjlyievmhenhkf4gq.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   x_2 => relu
#   x_3 => _unsafe_index_2, _unsafe_index_3
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu, [None, None, %sub_5, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %sub_5]), kwargs = {})
triton_poi_fused_reflection_pad2d_relu_2 = async_compile.triton('triton_poi_fused_reflection_pad2d_relu_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_relu_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 278784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = ((xindex // 66) % 66)
    x2 = xindex // 4356
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-1) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5m/c5mqex2g6juim5k7hd4vpsjbxa5kjpuxj7zev3lyf6uf22rjhwej.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   x_5 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %le_297 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_3 = async_compile.triton('triton_poi_fused_relu_threshold_backward_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/yo/cyoquxjbnew3xjmgj5thfdaqfxa7yct3nj3szcwgkrxhpwdaholv.py
# Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   x_5 => relu_1
#   x_6 => _unsafe_index_4, _unsafe_index_5
# Graph fragment:
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_1, [None, None, %sub_9, None]), kwargs = {})
#   %_unsafe_index_5 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_4, [None, None, None, %sub_9]), kwargs = {})
triton_poi_fused_reflection_pad2d_relu_4 = async_compile.triton('triton_poi_fused_reflection_pad2d_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_relu_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 34)
    x1 = ((xindex // 34) % 34)
    x2 = xindex // 1156
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e7/ce7web22m46xrgb3vyecbfpxpjn56xl6566vlaths77qoyxt6tww.py
# Topologically Sorted Source Nodes: [x_5, input_1, x_11], Original ATen: [aten.relu, aten.add, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_1 => add
#   x_11 => _unsafe_index_8, _unsafe_index_9
#   x_5 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %relu_1), kwargs = {})
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add, [None, None, %sub_9, None]), kwargs = {})
#   %_unsafe_index_9 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_8, [None, None, None, %sub_9]), kwargs = {})
triton_poi_fused_add_reflection_pad2d_relu_5 = async_compile.triton('triton_poi_fused_add_reflection_pad2d_relu_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_reflection_pad2d_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_reflection_pad2d_relu_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 34)
    x1 = ((xindex // 34) % 34)
    x2 = xindex // 1156
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ks/ckslk4zjqezkoojs5664tp6fldupboipooehfstzpav7su224isp.py
# Topologically Sorted Source Nodes: [x_5, input_1, input_2, x_16], Original ATen: [aten.relu, aten.add, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_1 => add
#   input_2 => add_1
#   x_16 => _unsafe_index_12, _unsafe_index_13
#   x_5 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %relu_1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add), kwargs = {})
#   %_unsafe_index_12 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1, [None, None, %sub_9, None]), kwargs = {})
#   %_unsafe_index_13 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_12, [None, None, None, %sub_9]), kwargs = {})
triton_poi_fused_add_reflection_pad2d_relu_6 = async_compile.triton('triton_poi_fused_add_reflection_pad2d_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_reflection_pad2d_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_reflection_pad2d_relu_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 34)
    x1 = ((xindex // 34) % 34)
    x2 = xindex // 1156
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp1 + tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a4/ca47s6ogqnsc56ojzurcuxjtaipcnzszsjjbphxnzfc4esplgzov.py
# Topologically Sorted Source Nodes: [x_5, input_1, input_2, input_3, x_21], Original ATen: [aten.relu, aten.add, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_1 => add
#   input_2 => add_1
#   input_3 => add_2
#   x_21 => _unsafe_index_16, _unsafe_index_17
#   x_5 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %relu_1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %add_1), kwargs = {})
#   %_unsafe_index_16 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_2, [None, None, %sub_9, None]), kwargs = {})
#   %_unsafe_index_17 : [num_users=3] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_16, [None, None, None, %sub_9]), kwargs = {})
triton_poi_fused_add_reflection_pad2d_relu_7 = async_compile.triton('triton_poi_fused_add_reflection_pad2d_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_reflection_pad2d_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_reflection_pad2d_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 34)
    x1 = ((xindex // 34) % 34)
    x2 = xindex // 1156
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 + tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kb/ckbtswocooesijl5pgkdrzwv37ezuhr2d6ho3mnoydpnyvovps5o.py
# Topologically Sorted Source Nodes: [x_23, x_24], Original ATen: [aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   x_23 => relu_5
#   x_24 => _unsafe_index_18, _unsafe_index_19
# Graph fragment:
#   %relu_5 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %_unsafe_index_18 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_5, [None, None, %sub_37, None]), kwargs = {})
#   %_unsafe_index_19 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_18, [None, None, None, %sub_37]), kwargs = {})
triton_poi_fused_reflection_pad2d_relu_8 = async_compile.triton('triton_poi_fused_reflection_pad2d_relu_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_relu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 18)
    x1 = ((xindex // 18) % 18)
    x2 = xindex // 324
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (255 + ((-1)*tl_math.abs((-15) + tl_math.abs((-1) + x0))) + ((-16)*tl_math.abs((-15) + tl_math.abs((-1) + x1))) + 256*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/du/cdum6l5p64h5noot6656dth2btf4cquzhp5vj76kkl5w7hqp2t5x.py
# Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   x_26 => relu_6
# Graph fragment:
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %le_148 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_9 = async_compile.triton('triton_poi_fused_relu_threshold_backward_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/tr/ctr3txhgx3kgo7ozqaalmqe2wjuyz47lmoe7bwuvpxqnb5duw4co.py
# Topologically Sorted Source Nodes: [x_23, input_4, x_29], Original ATen: [aten.relu, aten.add, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_4 => add_3
#   x_23 => relu_5
#   x_29 => _unsafe_index_22, _unsafe_index_23
# Graph fragment:
#   %relu_5 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %relu_5), kwargs = {})
#   %_unsafe_index_22 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_3, [None, None, %sub_37, None]), kwargs = {})
#   %_unsafe_index_23 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_22, [None, None, None, %sub_37]), kwargs = {})
triton_poi_fused_add_reflection_pad2d_relu_10 = async_compile.triton('triton_poi_fused_add_reflection_pad2d_relu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_reflection_pad2d_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_reflection_pad2d_relu_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 18)
    x1 = ((xindex // 18) % 18)
    x2 = xindex // 324
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (255 + ((-1)*tl_math.abs((-15) + tl_math.abs((-1) + x0))) + ((-16)*tl_math.abs((-15) + tl_math.abs((-1) + x1))) + 256*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (255 + ((-1)*tl_math.abs((-15) + tl_math.abs((-1) + x0))) + ((-16)*tl_math.abs((-15) + tl_math.abs((-1) + x1))) + 256*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/im/cimhpsyvcjcry2z6oxiw6jzgndhdqtxv3vvxiutsvltyafnqo5cy.py
# Topologically Sorted Source Nodes: [x_23, input_4, input_5], Original ATen: [aten.relu, aten.add, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_4 => add_3
#   input_5 => add_4
#   x_23 => relu_5
# Graph fragment:
#   %relu_5 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %relu_5), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %add_3), kwargs = {})
#   %le_167 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_5, 0), kwargs = {})
triton_poi_fused_add_relu_threshold_backward_11 = async_compile.triton('triton_poi_fused_add_relu_threshold_backward_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_relu_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_relu_threshold_backward_11(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp1 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = 0.0
    tmp8 = tmp4 <= tmp7
    tl.store(in_out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/jn/cjnpxdnz676obnzvqrdggdldowpv4r4zi5z5ut3rsietk22xekqd.py
# Topologically Sorted Source Nodes: [x_36, input_6, input_7], Original ATen: [aten.relu, aten.add, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_6 => add_5
#   input_7 => add_6
#   x_36 => relu_8
# Graph fragment:
#   %relu_8 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_15, %relu_8), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_5), kwargs = {})
#   %le_74 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
triton_poi_fused_add_relu_threshold_backward_12 = async_compile.triton('triton_poi_fused_add_relu_threshold_backward_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_relu_threshold_backward_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_relu_threshold_backward_12(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp1 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = 0.0
    tmp8 = tmp4 <= tmp7
    tl.store(in_out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (16, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_3, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_4, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_6, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_7, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_9, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_10, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_11, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_12, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_13, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_14, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_15, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_16, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_17, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_18, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_19, (32, 32, 3, 3), (288, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 3, 70, 70), (14700, 4900, 70, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_0.run(primals_1, buf0, 58800, grid=grid(58800), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf47 = empty_strided_cuda((4, 16, 64, 64), (65536, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_1.run(buf1, buf47, 262144, grid=grid(262144), stream=stream0)
        buf2 = empty_strided_cuda((4, 16, 66, 66), (69696, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_2.run(buf1, buf2, 278784, grid=grid(278784), stream=stream0)
        del buf1
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_3, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf46 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_3.run(buf3, buf46, 131072, grid=grid(131072), stream=stream0)
        buf4 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_4.run(buf3, buf4, 147968, grid=grid(147968), stream=stream0)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf45 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_3.run(buf5, buf45, 131072, grid=grid(131072), stream=stream0)
        buf6 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_4.run(buf5, buf6, 147968, grid=grid(147968), stream=stream0)
        del buf5
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf8 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, input_1, x_11], Original ATen: [aten.relu, aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_reflection_pad2d_relu_5.run(buf7, buf3, buf8, 147968, grid=grid(147968), stream=stream0)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf44 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_3.run(buf9, buf44, 131072, grid=grid(131072), stream=stream0)
        buf10 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_4.run(buf9, buf10, 147968, grid=grid(147968), stream=stream0)
        del buf9
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf12 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, input_1, input_2, x_16], Original ATen: [aten.relu, aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_reflection_pad2d_relu_6.run(buf11, buf7, buf3, buf12, 147968, grid=grid(147968), stream=stream0)
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf43 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_3.run(buf13, buf43, 131072, grid=grid(131072), stream=stream0)
        buf14 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_18, x_19], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_4.run(buf13, buf14, 147968, grid=grid(147968), stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf16 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, input_1, input_2, input_3, x_21], Original ATen: [aten.relu, aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_reflection_pad2d_relu_7.run(buf15, buf11, buf7, buf3, buf16, 147968, grid=grid(147968), stream=stream0)
        del buf11
        del buf15
        del buf3
        del buf7
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_10, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf18 = empty_strided_cuda((4, 32, 18, 18), (10368, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_23, x_24], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_8.run(buf17, buf18, 41472, grid=grid(41472), stream=stream0)
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf41 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_9.run(buf19, buf41, 32768, grid=grid(32768), stream=stream0)
        buf20 = empty_strided_cuda((4, 32, 18, 18), (10368, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_26, x_27], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_8.run(buf19, buf20, 41472, grid=grid(41472), stream=stream0)
        del buf19
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf22 = empty_strided_cuda((4, 32, 18, 18), (10368, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_23, input_4, x_29], Original ATen: [aten.relu, aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_reflection_pad2d_relu_10.run(buf21, buf17, buf22, 41472, grid=grid(41472), stream=stream0)
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf40 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_9.run(buf23, buf40, 32768, grid=grid(32768), stream=stream0)
        buf24 = empty_strided_cuda((4, 32, 18, 18), (10368, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_31, x_32], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_8.run(buf23, buf24, 41472, grid=grid(41472), stream=stream0)
        del buf23
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf26 = buf25; del buf25  # reuse
        buf42 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_23, input_4, input_5], Original ATen: [aten.relu, aten.add, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_relu_threshold_backward_11.run(buf26, buf21, buf17, buf42, 32768, grid=grid(32768), stream=stream0)
        del buf17
        del buf21
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf16, primals_15, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf28 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_36, x_37], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_4.run(buf27, buf28, 147968, grid=grid(147968), stream=stream0)
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf38 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_3.run(buf29, buf38, 131072, grid=grid(131072), stream=stream0)
        buf30 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_39, x_40], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_4.run(buf29, buf30, 147968, grid=grid(147968), stream=stream0)
        del buf29
        # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf32 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_36, input_6, x_42], Original ATen: [aten.relu, aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_reflection_pad2d_relu_5.run(buf31, buf27, buf32, 147968, grid=grid(147968), stream=stream0)
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf37 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_3.run(buf33, buf37, 131072, grid=grid(131072), stream=stream0)
        buf34 = empty_strided_cuda((4, 32, 34, 34), (36992, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_44, x_45], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_4.run(buf33, buf34, 147968, grid=grid(147968), stream=stream0)
        del buf33
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf36 = buf35; del buf35  # reuse
        buf39 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_36, input_6, input_7], Original ATen: [aten.relu, aten.add, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_relu_threshold_backward_12.run(buf36, buf31, buf27, buf39, 131072, grid=grid(131072), stream=stream0)
        del buf27
        del buf31
    return (buf26, buf36, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, buf0, buf2, buf4, buf6, buf8, buf10, buf12, buf14, buf16, buf18, buf20, buf22, buf24, buf28, buf30, buf32, buf34, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
