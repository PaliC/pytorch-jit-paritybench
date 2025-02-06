# AOT ID: ['27_forward']
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


# kernel path: inductor_cache/hz/chzs7s55cdsd6tvv4mppv7r7uxxhi6watv2txdgtqt6t2t5pykhy.py
# Topologically Sorted Source Nodes: [mv], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv => mul, sum_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %primals_6), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
triton_per_fused_mv_0 = async_compile.triton('triton_per_fused_mv_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mv_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mv_0(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 36*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dp/cdpzzdcwfwjz7z4fzsl43y6kgdmso3rqf73bukxczmgok3tpeb4a.py
# Topologically Sorted Source Nodes: [sigma], Original ATen: [aten.dot]
# Source node to ATen node mapping:
#   sigma => mul_1, sum_2
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %sum_1), kwargs = {})
#   %sum_2 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul_1,), kwargs = {})
triton_poi_fused_dot_1 = async_compile.triton('triton_poi_fused_dot_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_dot_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_dot_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (1))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (2))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp13 = tl.load(in_ptr1 + (2))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (3))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (3))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp4 = tmp1 * tmp3
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 + tmp9
    tmp15 = tmp12 * tmp14
    tmp16 = tmp10 + tmp15
    tmp21 = tmp18 * tmp20
    tmp22 = tmp16 + tmp21
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/kx/ckxggvytgsucoixnoyzgxoa5eyf3bbac7csuxbgmoud3p5m7fdc6.py
# Topologically Sorted Source Nodes: [weight], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight => div
# Graph fragment:
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_4, %sum_2), kwargs = {})
triton_poi_fused_div_2 = async_compile.triton('triton_poi_fused_div_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zm/czmfppxhrh6r25genoezot5otdr6dv5nn2d3qmmrekomc3zkul7t.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %div, %primals_7, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gw/cgwhphfihk2sll5hyfoqmgr7sqrc2esjkycvznwozgm7v7i7ccpm.py
# Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_3 => convolution_2
#   input_4 => gt_1, mul_5, where_1
# Graph fragment:
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_12, %div_1, %primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, 0.2), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_2, %mul_5), kwargs = {})
triton_poi_fused_convolution_leaky_relu_4 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dc/cdcogacljmkjnnvqcivov6myzwssblmelwgd5nnl5tdsah6ag2gx.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_10 => convert_element_type_3
# Graph fragment:
#   %convert_element_type_3 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.int64), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vu/cvurbfrktafldwiqjvlyh7nl43nsaasas7kq5bwb4bcunkcnglxi.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   input_10 => add_2, clamp_max
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_3, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_2, 3), kwargs = {})
triton_poi_fused_add_clamp_6 = async_compile.triton('triton_poi_fused_add_clamp_6', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n6/cn6ys5hnzmijgce5gcrggdm3hiz5nzg5yc7653j4xexfjpyav3pp.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   input_10 => add_1, clamp_max_2, clamp_min, clamp_min_2, convert_element_type_2, iota, mul_14, sub, sub_2
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 0.5), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_14, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_5), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_7 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pg/cpgmghyp3pelara5zqnvpmoa3pz4xrbdbkwfwcemb3wj4je3dylj.py
# Topologically Sorted Source Nodes: [input_2, input_9, mul, out, input_10], Original ATen: [aten.leaky_relu, aten.sigmoid, aten.mul, aten.add, aten._unsafe_index, aten.sub]
# Source node to ATen node mapping:
#   input_10 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_5, add_6, add_7, mul_16, mul_17, mul_18, sub_3, sub_4, sub_6
#   input_2 => gt, mul_2, where
#   input_9 => sigmoid
#   mul => mul_13
#   out => add
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.04), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_1, %mul_2), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_5,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %convolution_3), kwargs = {})
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %sigmoid), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add, [None, None, %convert_element_type_3, %convert_element_type_5]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add, [None, None, %convert_element_type_3, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add, [None, None, %clamp_max, %convert_element_type_5]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_16), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_17), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %add_5), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_18), kwargs = {})
triton_poi_fused__unsafe_index_add_leaky_relu_mul_sigmoid_sub_8 = async_compile.triton('triton_poi_fused__unsafe_index_add_leaky_relu_mul_sigmoid_sub_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_leaky_relu_mul_sigmoid_sub_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_leaky_relu_mul_sigmoid_sub_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.04
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.load(in_ptr3 + (tmp8 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp16 = tmp14 * tmp15
    tmp17 = tl.load(in_ptr4 + (tmp8 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20 + tmp1
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tl.load(in_ptr2 + (tmp8 + 4*tmp23 + 16*x2), xmask, eviction_policy='evict_last')
    tmp25 = tmp24 > tmp10
    tmp26 = tmp24 * tmp12
    tmp27 = tl.where(tmp25, tmp24, tmp26)
    tmp28 = tl.load(in_ptr3 + (tmp8 + 4*tmp23 + 16*x2), xmask, eviction_policy='evict_last')
    tmp29 = tmp27 * tmp28
    tmp30 = tl.load(in_ptr4 + (tmp8 + 4*tmp23 + 16*x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.sigmoid(tmp30)
    tmp32 = tmp29 + tmp31
    tmp34 = tmp33 + tmp1
    tmp35 = tmp33 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp33)
    tmp37 = tl.load(in_ptr2 + (tmp36 + 4*tmp23 + 16*x2), xmask, eviction_policy='evict_last')
    tmp38 = tmp37 > tmp10
    tmp39 = tmp37 * tmp12
    tmp40 = tl.where(tmp38, tmp37, tmp39)
    tmp41 = tl.load(in_ptr3 + (tmp36 + 4*tmp23 + 16*x2), xmask, eviction_policy='evict_last')
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr4 + (tmp36 + 4*tmp23 + 16*x2), xmask, eviction_policy='evict_last')
    tmp44 = tl.sigmoid(tmp43)
    tmp45 = tmp42 + tmp44
    tmp46 = tmp45 - tmp32
    tmp48 = tmp46 * tmp47
    tmp49 = tmp32 + tmp48
    tmp50 = tl.load(in_ptr2 + (tmp36 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp51 = tmp50 > tmp10
    tmp52 = tmp50 * tmp12
    tmp53 = tl.where(tmp51, tmp50, tmp52)
    tmp54 = tl.load(in_ptr3 + (tmp36 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp55 = tmp53 * tmp54
    tmp56 = tl.load(in_ptr4 + (tmp36 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp57 = tl.sigmoid(tmp56)
    tmp58 = tmp55 + tmp57
    tmp59 = tmp58 - tmp19
    tmp60 = tmp59 * tmp47
    tmp61 = tmp19 + tmp60
    tmp62 = tmp61 - tmp49
    tmp64 = tmp62 * tmp63
    tmp65 = tmp49 + tmp64
    tl.store(in_out_ptr0 + (x3), tmp65, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yv/cyvz7u7sj5srwmnwnnnjfybu4xgvzdq2mn5anhjf2bc6sx4esiyv.py
# Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   input_11 => convolution_6
#   input_12 => gt_3, mul_21, where_3
# Graph fragment:
#   %convolution_6 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_7, %div_5, %primals_28, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_6, 0), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_6, 0.2), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %convolution_6, %mul_21), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_3, 0), kwargs = {})
triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_9 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_9(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = tmp7 > tmp3
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28 = args
    args.clear()
    assert_size_stride(primals_1, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_4, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (36, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (36, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_13, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_14, (4, ), (1, ))
    assert_size_stride(primals_15, (36, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (36, ), (1, ))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_22, (4, ), (1, ))
    assert_size_stride(primals_23, (36, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_26, (4, ), (1, ))
    assert_size_stride(primals_27, (36, ), (1, ))
    assert_size_stride(primals_28, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 3, 3), (9, 9, 3, 1), torch.float32)
        buf0.copy_(primals_1, False)
        del primals_1
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_2, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf1, (4, 4, 4, 4), (64, 16, 4, 1))
        del buf0
        del primals_2
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_4, primals_6, buf2, 4, 36, grid=grid(4), stream=stream0)
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_5, buf2, buf3, 1, grid=grid(1), stream=stream0)
        buf4 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_4, buf3, buf4, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf1, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 4, 4, 4), (64, 16, 4, 1))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf6, primals_7, 256, grid=grid(256), stream=stream0)
        del primals_7
        buf7 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [mv_1], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_8, primals_10, buf7, 4, 36, grid=grid(4), stream=stream0)
        buf8 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_1], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_9, buf7, buf8, 1, grid=grid(1), stream=stream0)
        buf9 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_1], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_8, buf8, buf9, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(primals_12, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 4, 4, 4), (64, 16, 4, 1))
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_4.run(buf11, primals_11, 256, grid=grid(256), stream=stream0)
        del primals_11
        buf12 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [mv_2], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_13, primals_15, buf12, 4, 36, grid=grid(4), stream=stream0)
        buf13 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_2], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_14, buf12, buf13, 1, grid=grid(1), stream=stream0)
        buf14 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_2], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_13, buf13, buf14, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf11, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 4, 4, 4), (64, 16, 4, 1))
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf16, primals_16, 256, grid=grid(256), stream=stream0)
        del primals_16
        buf17 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [mv_3], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_17, primals_19, buf17, 4, 36, grid=grid(4), stream=stream0)
        buf18 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_3], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_18, buf17, buf18, 1, grid=grid(1), stream=stream0)
        buf19 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_3], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_17, buf18, buf19, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(primals_12, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 4, 4, 4), (64, 16, 4, 1))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_4.run(buf21, primals_20, 256, grid=grid(256), stream=stream0)
        del primals_20
        buf22 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [mv_4], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_21, primals_23, buf22, 4, 36, grid=grid(4), stream=stream0)
        buf23 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_4], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_22, buf22, buf23, 1, grid=grid(1), stream=stream0)
        buf24 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_4], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_21, buf23, buf24, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf21, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 4, 4, 4), (64, 16, 4, 1))
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf26, primals_24, 256, grid=grid(256), stream=stream0)
        del primals_24
        buf27 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf27, 8, grid=grid(8), stream=stream0)
        buf28 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_6.run(buf28, 8, grid=grid(8), stream=stream0)
        buf29 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf29, 8, grid=grid(8), stream=stream0)
        buf30 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_6.run(buf30, 8, grid=grid(8), stream=stream0)
        buf33 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_7.run(buf33, 8, grid=grid(8), stream=stream0)
        buf35 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_7.run(buf35, 8, grid=grid(8), stream=stream0)
        buf31 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        buf34 = buf31; del buf31  # reuse
        buf37 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_9, mul, out, input_10], Original ATen: [aten.leaky_relu, aten.sigmoid, aten.mul, aten.add, aten._unsafe_index, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_leaky_relu_mul_sigmoid_sub_8.run(buf37, buf28, buf29, buf6, buf16, buf26, buf27, buf30, buf33, buf35, 1024, grid=grid(1024), stream=stream0)
        buf38 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [mv_5], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_25, primals_27, buf38, 4, 36, grid=grid(4), stream=stream0)
        buf39 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_5], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_26, buf38, buf39, 1, grid=grid(1), stream=stream0)
        del buf38
        buf40 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_5], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_25, buf39, buf40, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf37, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 4, 8, 8), (256, 64, 8, 1))
        buf42 = buf41; del buf41  # reuse
        buf43 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_9.run(buf42, primals_28, buf43, 1024, grid=grid(1024), stream=stream0)
        del primals_28
    return (buf42, buf4, buf9, buf14, buf19, buf24, buf40, primals_4, primals_5, primals_6, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_25, primals_26, primals_27, buf1, buf3, buf4, buf6, buf8, buf9, buf11, buf13, buf14, buf16, buf18, buf19, buf21, buf23, buf24, buf26, buf27, buf28, buf29, buf30, buf33, buf35, buf37, buf39, buf40, buf43, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
