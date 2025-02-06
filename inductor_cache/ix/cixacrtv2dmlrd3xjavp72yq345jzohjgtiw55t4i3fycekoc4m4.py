# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/6u/c6u47aqzgu2rszryayemnnsqak3c7ndva4yn7tbav3hmzwcrrmre.py
# Topologically Sorted Source Nodes: [pad], Original ATen: [aten.replication_pad2d]
# Source node to ATen node mapping:
#   pad => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_3, [None, None, %clamp_max, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %clamp_max]), kwargs = {})
triton_poi_fused_replication_pad2d_0 = async_compile.triton('triton_poi_fused_replication_pad2d_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_replication_pad2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_replication_pad2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = ((xindex // 66) % 66)
    x2 = xindex // 4356
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (64*((63) * ((63) <= (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))) + (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) < (63))) + 4096*x2 + ((63) * ((63) <= (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))) + (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) < (63)))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p3/cp3skxarhu3tctxbufhhyxwhdeaiadaygoleb63lxpmmxnijvzwr.py
# Topologically Sorted Source Nodes: [conv2d, x], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d => convolution
#   x => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_1, %primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_1 = async_compile.triton('triton_poi_fused_convolution_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/nm/cnmhtad5rl2t3caffarpbzaui7hrn6jugu7e3bz4rfqwheut76z6.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu, [2, 2], [2, 2]), kwargs = {})
triton_poi_fused_avg_pool2d_2 = async_compile.triton('triton_poi_fused_avg_pool2d_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/ql/cqlpccp2hrihdqmapk22inzk63ahau5aflgp6pylzczksl7npo2l.py
# Topologically Sorted Source Nodes: [pad_1], Original ATen: [aten.replication_pad2d]
# Source node to ATen node mapping:
#   pad_1 => _unsafe_index_2, _unsafe_index_3
# Graph fragment:
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%avg_pool2d, [None, None, %clamp_max_2, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %clamp_max_2]), kwargs = {})
triton_poi_fused_replication_pad2d_3 = async_compile.triton('triton_poi_fused_replication_pad2d_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_replication_pad2d_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_replication_pad2d_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 82944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 36)
    x1 = ((xindex // 36) % 36)
    x2 = xindex // 1296
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (32*((31) * ((31) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < (31))) + 1024*x2 + ((31) * ((31) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (31)))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2b/c2b54a6uhjqlu5j2kowaws3d3ulufbulx4ffx5chnunvb2xe5njl.py
# Topologically Sorted Source Nodes: [conv2d_1, x_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
#   x_2 => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_3, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/ms/cmsjl2ct4fxlpr2dzgbqmkggq2xbleunlbovmat2eebpk4sutbr3.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_3 => avg_pool2d_1
# Graph fragment:
#   %avg_pool2d_1 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_1, [2, 2], [2, 2]), kwargs = {})
triton_poi_fused_avg_pool2d_5 = async_compile.triton('triton_poi_fused_avg_pool2d_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/kn/cknmwqiwv2mai7hyjpa3y7axfgdiddoeffuetexoxxxtycvniake.py
# Topologically Sorted Source Nodes: [pad_2], Original ATen: [aten.replication_pad2d]
# Source node to ATen node mapping:
#   pad_2 => _unsafe_index_4, _unsafe_index_5
# Graph fragment:
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%avg_pool2d_1, [None, None, %clamp_max_4, None]), kwargs = {})
#   %_unsafe_index_5 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_4, [None, None, None, %clamp_max_4]), kwargs = {})
triton_poi_fused_replication_pad2d_6 = async_compile.triton('triton_poi_fused_replication_pad2d_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_replication_pad2d_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_replication_pad2d_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 20)
    x1 = ((xindex // 20) % 20)
    x2 = xindex // 400
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (16*((15) * ((15) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < (15))) + 256*x2 + ((15) * ((15) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (15)))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/li/cli3a62dhdrbs3ably3okla3eonzex2c7uh6rxeyzq742hrqgpfd.py
# Topologically Sorted Source Nodes: [conv2d_2, x_4], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d_2 => convolution_2
#   x_4 => relu_2
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_5, %primals_6, %primals_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_convolution_relu_7 = async_compile.triton('triton_poi_fused_convolution_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/6q/c6qghzhutebkh5jllydjqd5rj4kec45hh22oaqhbzq2tsqmonh35.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_5 => avg_pool2d_2
# Graph fragment:
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_2, [2, 2], [2, 2]), kwargs = {})
triton_poi_fused_avg_pool2d_8 = async_compile.triton('triton_poi_fused_avg_pool2d_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/y4/cy4e4ndmjsfhbzxljcm6uh4dbqvy2izsf7bh5sbx5iolszdj65we.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   x_6 => addmm
# Graph fragment:
#   %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_9, %select_1, %permute), kwargs = {})
triton_poi_fused_addmm_9 = async_compile.triton('triton_poi_fused_addmm_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (64*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (4, 1, 64, 64), (4096, 4096, 64, 1))
    assert_size_stride(primals_4, (32, 16, 5, 5), (400, 25, 5, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (64, 32, 5, 5), (800, 25, 5, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (10, 64), (64, 1))
    assert_size_stride(primals_9, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 66, 66), (4356, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad], Original ATen: [aten.replication_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_replication_pad2d_0.run(primals_3, buf0, 17424, grid=grid(17424), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [conv2d, x], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf2, primals_2, 262144, grid=grid(262144), stream=stream0)
        del primals_2
        buf3 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_2.run(buf2, buf3, 65536, grid=grid(65536), stream=stream0)
        buf4 = empty_strided_cuda((4, 16, 36, 36), (20736, 1296, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_1], Original ATen: [aten.replication_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_replication_pad2d_3.run(buf3, buf4, 82944, grid=grid(82944), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1, x_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_4.run(buf6, primals_5, 131072, grid=grid(131072), stream=stream0)
        del primals_5
        buf7 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_5.run(buf6, buf7, 32768, grid=grid(32768), stream=stream0)
        buf8 = empty_strided_cuda((4, 32, 20, 20), (12800, 400, 20, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_2], Original ATen: [aten.replication_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_replication_pad2d_6.run(buf7, buf8, 51200, grid=grid(51200), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [conv2d_2, x_4], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_7.run(buf10, primals_7, 65536, grid=grid(65536), stream=stream0)
        del primals_7
        buf11 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_8.run(buf10, buf11, 16384, grid=grid(16384), stream=stream0)
        buf12 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_9.run(buf11, buf12, 256, grid=grid(256), stream=stream0)
        buf13 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf12, reinterpret_tensor(primals_8, (64, 10), (1, 64), 0), alpha=1, beta=1, out=buf13)
        del buf12
        del primals_9
    return (buf13, primals_1, primals_4, primals_6, buf0, buf2, buf3, buf4, buf6, buf7, buf8, buf10, reinterpret_tensor(buf11, (4, 64), (4096, 64), 0), primals_8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 1, 64, 64), (4096, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 32, 5, 5), (800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((10, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
