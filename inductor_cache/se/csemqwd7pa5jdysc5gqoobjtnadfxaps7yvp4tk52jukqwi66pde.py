# AOT ID: ['5_forward']
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


# kernel path: inductor_cache/5b/c5bc35vvmpdiog76nlgag7q2k2jp7pz5ds74af276kiset36ecin.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 3), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 3)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ql/cqlennvxv5vibkwzf25pzhk5j4iauoazz7iqgkrx67jki55e6hrz.py
# Topologically Sorted Source Nodes: [out_1, x1], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out_1 => convolution_1
#   x1 => relu
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
triton_poi_fused_convolution_relu_1 = async_compile.triton('triton_poi_fused_convolution_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/mu/cmuuluuh2eunsabdfp2w4x3swzhtmqb6n56mqo757uz73mnjbt4p.py
# Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_2 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_6, %primals_7, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/zq/czqqckm4q7nljh4wcwjpos5upwsnvnelqx52akrnaqctzizjsin2.py
# Topologically Sorted Source Nodes: [out_7, x4], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_7 => convolution_7
#   x4 => relu_3
# Graph fragment:
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_6, %primals_16, %primals_17, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
triton_poi_fused_convolution_relu_threshold_backward_3 = async_compile.triton('triton_poi_fused_convolution_relu_threshold_backward_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/nj/cnj5ffai3gtxtklhgyj6qywlez7uzkvuaukclglkjxp2qpe6naf5.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_2, %relu_3], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 64)
    x0 = (xindex % 4096)
    x2 = xindex // 262144
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 131072*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 64, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-32) + x1) + 131072*x2), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-32) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/di/cdirg3tksrd7wtchhezwcyzbivynnmbdmh5ch4dc4a4iqvzgv6g4.py
# Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_8 => convolution_8
# Graph fragment:
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_18, %primals_19, [1, 1], [1, 1], [1, 1], False, [0, 0], 64), kwargs = {})
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/uq/cuqmst4repmt6wuko3bzlbyyeqyzub6xz5nzhnucbwymg3e3wds5.py
# Topologically Sorted Source Nodes: [out_13, x_r, pow_1, sub, mul, x, pow_2, sub_1, mul_1, x_1, pow_3, sub_2, mul_2, x_2, pow_4, sub_3, mul_3, enhance_image_1, pow_5, sub_4, mul_4, x_3, pow_6, sub_5, mul_5, x_4, pow_7, sub_6, mul_6, x_5, pow_8, sub_7, mul_7, enhance_image], Original ATen: [aten.convolution, aten.tanh, aten.pow, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   enhance_image => add_7
#   enhance_image_1 => add_3
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   out_13 => convolution_13
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_6 => pow_6
#   pow_7 => pow_7
#   pow_8 => pow_8
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_6 => sub_6
#   sub_7 => sub_7
#   x => add
#   x_1 => add_1
#   x_2 => add_2
#   x_3 => add_4
#   x_4 => add_5
#   x_5 => add_6
#   x_r => tanh
# Graph fragment:
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_12, %primals_28, %primals_29, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %tanh : [num_users=9] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_13,), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_1, %primals_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %sub), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %mul), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_2, %add), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %sub_1), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_1), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1, 2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_3, %add_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %sub_2), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_2), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_2, 2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_4, %add_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %sub_3), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_3), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_3, 2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_5, %add_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %sub_4), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_4), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_4, 2), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_6, %add_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %sub_5), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_5), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_7, %add_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %sub_6), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_6), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6, 2), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_8, %add_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %sub_7), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mul_7), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_sub_tanh_6 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_sub_tanh_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_sub_tanh_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_sub_tanh_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 3)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tmp5 = tmp4 * tmp4
    tmp6 = tmp5 - tmp4
    tmp7 = tmp3 * tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tmp9 - tmp8
    tmp11 = tmp3 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tmp13 - tmp12
    tmp15 = tmp3 * tmp14
    tmp16 = tmp12 + tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp17 - tmp16
    tmp19 = tmp3 * tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tmp21 - tmp20
    tmp23 = tmp3 * tmp22
    tmp24 = tmp20 + tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tmp25 - tmp24
    tmp27 = tmp3 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tmp29 - tmp28
    tmp31 = tmp3 * tmp30
    tmp32 = tmp28 + tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tmp33 - tmp32
    tmp35 = tmp3 * tmp34
    tmp36 = tmp32 + tmp35
    tl.store(in_out_ptr0 + (x3), tmp3, None)
    tl.store(in_out_ptr1 + (x3), tmp36, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (3, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_3, (3, ), (1, ))
    assert_size_stride(primals_4, (32, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (3, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_29, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf0, (4, 3, 64, 64), (12288, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf1, primals_3, 49152, grid=grid(49152), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [out_1, x1], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf3, primals_5, 524288, grid=grid(524288), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf4, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf5, primals_7, 524288, grid=grid(524288), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [out_3, x2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf7, primals_9, 524288, grid=grid(524288), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf8, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf9, primals_11, 524288, grid=grid(524288), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [out_5, x3], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf11, primals_13, 524288, grid=grid(524288), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf12, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf13, primals_15, 524288, grid=grid(524288), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf33 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_7, x4], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_3.run(buf14, primals_17, buf33, 524288, grid=grid(524288), stream=stream0)
        buf15 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf11, buf14, primals_17, buf15, 1048576, grid=grid(1048576), stream=stream0)
        del buf14
        del primals_17
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf16, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_5.run(buf17, primals_19, 1048576, grid=grid(1048576), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf32 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_9, x5], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_3.run(buf18, primals_21, buf32, 524288, grid=grid(524288), stream=stream0)
        buf19 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf7, buf18, primals_21, buf19, 1048576, grid=grid(1048576), stream=stream0)
        del buf18
        del primals_21
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf20, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_5.run(buf21, primals_23, 1048576, grid=grid(1048576), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_24, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf31 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_11, x6], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_3.run(buf22, primals_25, buf31, 524288, grid=grid(524288), stream=stream0)
        buf23 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf3, buf22, primals_25, buf23, 1048576, grid=grid(1048576), stream=stream0)
        del buf22
        del primals_25
        # Topologically Sorted Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf24, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_5.run(buf25, primals_27, 1048576, grid=grid(1048576), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 3, 64, 64), (12288, 4096, 64, 1))
        buf27 = buf26; del buf26  # reuse
        buf28 = empty_strided_cuda((4, 3, 64, 64), (12288, 4096, 64, 1), torch.float32)
        buf29 = buf28; del buf28  # reuse
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [out_13, x_r, pow_1, sub, mul, x, pow_2, sub_1, mul_1, x_1, pow_3, sub_2, mul_2, x_2, pow_4, sub_3, mul_3, enhance_image_1, pow_5, sub_4, mul_4, x_3, pow_6, sub_5, mul_5, x_4, pow_7, sub_6, mul_6, x_5, pow_8, sub_7, mul_7, enhance_image], Original ATen: [aten.convolution, aten.tanh, aten.pow, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_sub_tanh_6.run(buf27, buf30, primals_29, primals_1, 49152, grid=grid(49152), stream=stream0)
        del primals_29
    return (buf30, buf27, primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf27, buf31, buf32, buf33, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((3, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((3, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
