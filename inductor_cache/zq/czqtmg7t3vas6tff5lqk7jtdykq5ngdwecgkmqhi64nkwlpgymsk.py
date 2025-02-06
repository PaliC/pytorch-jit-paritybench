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


# kernel path: inductor_cache/b3/cb3hjr3k67j7aszrq2yeg7ymlijoxof5oaehjz64xgqb74rvehqx.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_15 => _unsafe_index_10, _unsafe_index_11
# Graph fragment:
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_17, None]), kwargs = {})
#   %_unsafe_index_11 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_10, [None, None, None, %sub_17]), kwargs = {})
triton_poi_fused_reflection_pad2d_0 = async_compile.triton('triton_poi_fused_reflection_pad2d_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 69696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = ((xindex // 66) % 66)
    x2 = xindex // 4356
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-1) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nj/cnjhg3yng5thk4scwyxlqzccvouhjlwjfov34kr7cf3aykqw2qj4.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_1 => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad2d_1 = async_compile.triton('triton_poi_fused_reflection_pad2d_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 68)
    x1 = ((xindex // 68) % 68)
    x2 = xindex // 4624
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-2) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-2) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2h/c2hpnjatbvdj4lsspqb5mcb2vdkjkwax2linbsrdwqz3ij75j4dt.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_2 => convolution
#   input_3 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_2, %primals_3, [1, 1], [0, 0], [2, 2], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %le_21 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_convolution_relu_threshold_backward_2 = async_compile.triton('triton_poi_fused_convolution_relu_threshold_backward_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = 0.0
    tmp7 = tmp5 <= tmp6
    tl.store(out_ptr0 + (x0), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/jn/cjnamvnwlv5oumbj6zfzrn6wrbpxazofb2uycv4wdan6ifvoytrp.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_4 => _unsafe_index_2, _unsafe_index_3
# Graph fragment:
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_5, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %sub_5]), kwargs = {})
triton_poi_fused_reflection_pad2d_3 = async_compile.triton('triton_poi_fused_reflection_pad2d_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 82944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 72)
    x1 = ((xindex // 72) % 72)
    x2 = xindex // 5184
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-4) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-4) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bq/cbqotjupwlooqysfiwbsz4pveyqstffvoatzma4wvus4wy2lxdb7.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_7 => _unsafe_index_4, _unsafe_index_5
# Graph fragment:
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_9, None]), kwargs = {})
#   %_unsafe_index_5 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_4, [None, None, None, %sub_9]), kwargs = {})
triton_poi_fused_reflection_pad2d_4 = async_compile.triton('triton_poi_fused_reflection_pad2d_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 80)
    x1 = ((xindex // 80) % 80)
    x2 = xindex // 6400
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-8) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-8) + x1))) + 4096*x2), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/dz/cdz3n7gs5nw53xsmutnwtndzxppyidzx6r7enfyf2p4ermt2t3zn.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_10 => _unsafe_index_6, _unsafe_index_7
# Graph fragment:
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_13, None]), kwargs = {})
#   %_unsafe_index_7 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_6, [None, None, None, %sub_13]), kwargs = {})
triton_poi_fused_reflection_pad2d_5 = async_compile.triton('triton_poi_fused_reflection_pad2d_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 96)
    x2 = xindex // 9216
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-16) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-16) + x1))) + 4096*x2), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/64/c64ookuvl6oog3owgsk7l3zm5ow664xinncekx2ekyosauu3smy3.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu, %relu_1, %relu_2, %relu_3], 1), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 4)
    x0 = (xindex % 4096)
    x2 = xindex // 16384
    x3 = xindex
    tmp6 = tl.load(in_ptr1 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp18 = tl.load(in_ptr3 + (0))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp30 = tl.load(in_ptr5 + (0))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp41 = tl.load(in_ptr7 + (0))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp5 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 2, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tmp13 & tmp15
    tmp17 = tl.load(in_ptr2 + (x0 + 4096*x2), tmp16, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp17 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp16, tmp22, tmp23)
    tmp25 = tmp0 >= tmp14
    tmp26 = tl.full([1], 3, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr4 + (x0 + 4096*x2), tmp28, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp29 + tmp31
    tmp33 = tl.full([1], 0, tl.int32)
    tmp34 = triton_helpers.maximum(tmp33, tmp32)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp28, tmp34, tmp35)
    tmp37 = tmp0 >= tmp26
    tmp38 = tl.full([1], 4, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tl.load(in_ptr6 + (x0 + 4096*x2), tmp37, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp40 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp37, tmp45, tmp46)
    tmp48 = tl.where(tmp28, tmp36, tmp47)
    tmp49 = tl.where(tmp16, tmp24, tmp48)
    tmp50 = tl.where(tmp4, tmp12, tmp49)
    tl.store(out_ptr0 + (x3), tmp50, None)
''', device_str='cuda')


# kernel path: inductor_cache/xc/cxctr5ax3ww2whl24pse7udarb7qy5q5lhy7lx35wgcjvgfns5zh.py
# Topologically Sorted Source Nodes: [input_14, input_16, mean, std, std_1, sub, mul, truediv, feat, feat_1, mask, sub_2, mul_2, mul_3, add_1], Original ATen: [aten.convolution, aten.mean, aten.std, aten.add, aten.sub, aten.mul, aten.div, aten.sigmoid, aten.rsub]
# Source node to ATen node mapping:
#   add_1 => add_1
#   feat => sub_25
#   feat_1 => mul_1
#   input_14 => convolution_4
#   input_16 => convolution_5
#   mask => sigmoid
#   mean => mean
#   mul => mul
#   mul_2 => mul_2
#   mul_3 => mul_3
#   std => sqrt, var
#   std_1 => add
#   sub => sub_24
#   sub_2 => sub_26
#   truediv => div
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_9, %primals_10, %primals_11, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_5 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_11, %primals_12, %primals_13, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_5, [2, 3], True), kwargs = {})
#   %var : [num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%convolution_5, [2, 3]), kwargs = {correction: 1.0, keepdim: True})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%var,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt, 1e-09), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %mean), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, 2), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %add), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, 1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, 5), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_1,), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %sub_26), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_4, %sigmoid), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
triton_red_fused_add_convolution_div_mean_mul_rsub_sigmoid_std_sub_7 = async_compile.triton('triton_red_fused_add_convolution_div_mean_mul_rsub_sigmoid_std_sub_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_div_mean_mul_rsub_sigmoid_std_sub_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_div_mean_mul_rsub_sigmoid_std_sub_7(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = (xindex % 4)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + 4096*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r2 + 4096*x3), tmp2, rmask & xmask)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp9 = 4096.0
    tmp10 = tmp4 / tmp9
    tmp11 = 4095.0
    tmp12 = tmp7 / tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp10, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x3), tmp13, xmask)
    tmp15 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_out_ptr3 + (r2 + 4096*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2 + 4096*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_out_ptr0 + (r2 + 4096*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tmp14 + tmp15
        tmp19 = tmp18 - tmp10
        tmp20 = 2.0
        tmp21 = tmp19 * tmp20
        tmp22 = 1e-09
        tmp23 = tmp13 + tmp22
        tmp24 = tmp21 / tmp23
        tmp25 = 1.0
        tmp26 = tmp24 - tmp25
        tmp27 = 5.0
        tmp28 = tmp26 * tmp27
        tmp29 = tl.sigmoid(tmp28)
        tmp30 = tmp25 - tmp29
        tmp31 = tmp17 * tmp30
        tmp32 = tmp16 * tmp29
        tmp33 = tmp31 + tmp32
        tl.store(in_out_ptr3 + (r2 + 4096*x3), tmp16, rmask & xmask)
        tl.store(out_ptr0 + (r2 + 4096*x3), tmp33, rmask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(primals_2, (1, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_3, (1, ), (1, ))
    assert_size_stride(primals_4, (1, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_5, (1, ), (1, ))
    assert_size_stride(primals_6, (1, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_7, (1, ), (1, ))
    assert_size_stride(primals_8, (1, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_9, (1, ), (1, ))
    assert_size_stride(primals_10, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf12 = empty_strided_cuda((4, 4, 66, 66), (17424, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_0.run(primals_1, buf12, 69696, grid=grid(69696), stream=stream0)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf0 = empty_strided_cuda((4, 4, 68, 68), (18496, 4624, 68, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_1.run(primals_1, buf0, 73984, grid=grid(73984), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 1, 64, 64), (4096, 4096, 64, 1))
        buf25 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_2.run(buf1, primals_3, buf25, 16384, grid=grid(16384), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 72, 72), (20736, 5184, 72, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_3.run(primals_1, buf2, 82944, grid=grid(82944), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1), padding=(0, 0), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 1, 64, 64), (4096, 4096, 64, 1))
        buf24 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_2.run(buf3, primals_5, buf24, 16384, grid=grid(16384), stream=stream0)
        buf4 = empty_strided_cuda((4, 4, 80, 80), (25600, 6400, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_4.run(primals_1, buf4, 102400, grid=grid(102400), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_6, stride=(1, 1), padding=(0, 0), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 1, 64, 64), (4096, 4096, 64, 1))
        buf23 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_2.run(buf5, primals_7, buf23, 16384, grid=grid(16384), stream=stream0)
        buf6 = empty_strided_cuda((4, 4, 96, 96), (36864, 9216, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(primals_1, buf6, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_8, stride=(1, 1), padding=(0, 0), dilation=(16, 16), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 1, 64, 64), (4096, 4096, 64, 1))
        buf22 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_2.run(buf7, primals_9, buf22, 16384, grid=grid(16384), stream=stream0)
        buf8 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf1, primals_3, buf3, primals_5, buf5, primals_7, buf7, primals_9, buf8, 65536, grid=grid(65536), stream=stream0)
        del buf1
        del buf3
        del buf5
        del buf7
        del primals_3
        del primals_5
        del primals_7
        del primals_9
        buf9 = empty_strided_cuda((4, 4, 66, 66), (17424, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_0.run(buf8, buf9, 69696, grid=grid(69696), stream=stream0)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf18 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf16 = reinterpret_tensor(buf15, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf15  # reuse
        buf20 = reinterpret_tensor(buf18, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf18  # reuse
        buf11 = buf10; del buf10  # reuse
        buf21 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_14, input_16, mean, std, std_1, sub, mul, truediv, feat, feat_1, mask, sub_2, mul_2, mul_3, add_1], Original ATen: [aten.convolution, aten.mean, aten.std, aten.add, aten.sub, aten.mul, aten.div, aten.sigmoid, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_div_mean_mul_rsub_sigmoid_std_sub_7.run(buf14, buf16, buf20, buf11, primals_13, primals_11, primals_1, buf21, 16, 4096, grid=grid(16), stream=stream0)
        del primals_11
        del primals_13
    return (buf21, primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_12, buf0, buf2, buf4, buf6, buf9, buf11, buf12, buf14, buf16, buf20, buf22, buf23, buf24, buf25, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
