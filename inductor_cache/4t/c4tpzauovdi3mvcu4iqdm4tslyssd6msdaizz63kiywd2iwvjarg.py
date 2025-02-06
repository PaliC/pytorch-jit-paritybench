# AOT ID: ['13_forward']
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


# kernel path: inductor_cache/sg/csg5rm5h2gj33woe5g2dbfqm4ugdg3nluxgo46pr55gjiyf4cdps.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_4, %view], 2), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 6)
    x3 = xindex // 96
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 64*x3), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 6, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = x1 + 4*((-4) + x2)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp9 < tmp12
    tmp14 = tmp13 & tmp6
    tmp15 = tl.load(in_ptr1 + (x0 + 4*(x1 + 4*((-4) + x2)) + 16*x3), tmp14 & xmask, other=0.0)
    tmp16 = tmp9 >= tmp12
    tmp17 = tl.full([1], 8, tl.int64)
    tmp18 = tmp9 < tmp17
    tmp19 = tmp16 & tmp6
    tmp20 = tl.load(in_ptr2 + (x0 + 4*((-4) + x1 + 4*((-4) + x2)) + 16*x3), tmp19 & xmask, other=0.0)
    tmp21 = tl.where(tmp13, tmp15, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp6, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp5, tmp23)
    tl.store(out_ptr0 + (x5), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yx/cyxol3pxv24dlewq5mftvxo6rairn2he5tvg65jt65qyycebiap4.py
# Topologically Sorted Source Nodes: [p16], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   p16 => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_2), kwargs = {})
triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = (xindex % 256)
    x5 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x5), xmask)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x5), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t5/ct5b7xrgp36os76khanxiao7ewyas5ehgg7iasj7wdybubbhq43n.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_5,), kwargs = {})
triton_poi_fused_relu_2 = async_compile.triton('triton_poi_fused_relu_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 64
    x3 = (xindex % 64)
    x1 = ((xindex // 16) % 4)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + 64*(x2 // 4)), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(out_ptr0 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yv/cyvuh5eyuqsip7rb4nxa5g53ebyaear2qsczapt7afam52xsmdc4.py
# Topologically Sorted Source Nodes: [x_2, relu_1], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   relu_1 => relu_1
#   x_2 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_12, %primals_13, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
triton_poi_fused_convolution_relu_3 = async_compile.triton('triton_poi_fused_convolution_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/24/c24qws7s7c5qjivybjhlgyozipnug7v2dcshruh5ddk4wbves7iq.py
# Topologically Sorted Source Nodes: [x_3, adaptive_avg_pool2d], Original ATen: [aten.convolution, aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d => mean
#   x_3 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_14, %primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_4, [-1, -2], True), kwargs = {})
triton_per_fused_convolution_mean_4 = async_compile.triton('triton_per_fused_convolution_mean_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_mean_4(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 16*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 16.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/63/c63skfbxvqsn3rqeunjjbeo5clb6m2xnxxoqaubqazzsveei3dx5.py
# Topologically Sorted Source Nodes: [w_1, mul, x_4, relu_2], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.relu]
# Source node to ATen node mapping:
#   mul => mul
#   relu_2 => relu_2
#   w_1 => sigmoid
#   x_4 => add_2
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%unsqueeze_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_4, %sigmoid), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %view_5), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_2,), kwargs = {})
triton_poi_fused_add_mul_relu_sigmoid_5 = async_compile.triton('triton_poi_fused_add_mul_relu_sigmoid_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_relu_sigmoid_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_relu_sigmoid_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex // 16
    x2 = xindex // 64
    x5 = (xindex % 64)
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x5 + 64*(x2 // 4)), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp3 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6l/c6lkpr5b2tuiqfaverfkcoud4u3mrshqcyribdpy4bhghrkjnhnx.py
# Topologically Sorted Source Nodes: [w_3, mul_1, x_7], Original ATen: [aten.sigmoid, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_1 => mul_1
#   w_3 => sigmoid_1
#   x_7 => add_3
# Graph fragment:
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%unsqueeze_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_7, %sigmoid_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %add_2), kwargs = {})
triton_poi_fused_add_mul_sigmoid_6 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_6(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 4, 4, 4, 4), (256, 64, 16, 4, 1))
    assert_size_stride(primals_5, (4, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_10, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_17, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (1, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 6, 4, 4), (384, 96, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(primals_4, primals_3, primals_2, buf0, 1536, grid=grid(1536), stream=stream0)
        del primals_2
        del primals_3
        del primals_4
        # Topologically Sorted Source Nodes: [g], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(reinterpret_tensor(buf0, (16, 6, 4, 4), (96, 16, 4, 1), 0), primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (16, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(primals_9, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 4, 4, 4), (64, 16, 4, 1))
        buf3 = reinterpret_tensor(buf1, (4, 4, 4, 4, 4), (256, 64, 16, 4, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [p16], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_1.run(buf3, primals_1, primals_6, 1024, grid=grid(1024), stream=stream0)
        del primals_1
        del primals_6
        # Topologically Sorted Source Nodes: [g_1], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(reinterpret_tensor(buf3, (16, 4, 4, 4), (64, 16, 4, 1), 0), primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (16, 4, 4, 4), (64, 16, 4, 1))
        buf5 = empty_strided_cuda((16, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_2.run(buf2, primals_8, buf4, primals_11, buf5, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (16, 4, 4, 4), (64, 16, 4, 1))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_2, relu_1], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_3.run(buf7, primals_13, 1024, grid=grid(1024), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (16, 4, 4, 4), (64, 16, 4, 1))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((16, 4, 1, 1), (4, 1, 64, 64), torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_3, adaptive_avg_pool2d], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_mean_4.run(buf9, buf11, primals_15, 64, 16, grid=grid(64), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(reinterpret_tensor(buf11, (16, 1, 4), (4, 0, 1), 0), primals_16, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf12, (16, 1, 4), (4, 4, 1))
        buf13 = buf4; del buf4  # reuse
        buf14 = empty_strided_cuda((16, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [w_1, mul, x_4, relu_2], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_relu_sigmoid_5.run(buf13, buf9, buf12, buf2, primals_8, primals_11, buf14, 1024, grid=grid(1024), stream=stream0)
        del buf2
        del primals_11
        del primals_8
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (16, 4, 4, 4), (64, 16, 4, 1))
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_5, relu_3], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_3.run(buf16, primals_18, 1024, grid=grid(1024), stream=stream0)
        del primals_18
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (16, 4, 4, 4), (64, 16, 4, 1))
        buf18 = buf17; del buf17  # reuse
        buf19 = empty_strided_cuda((16, 4, 1, 1), (4, 1, 64, 64), torch.float32)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_6, adaptive_avg_pool2d_1], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_mean_4.run(buf18, buf20, primals_20, 64, 16, grid=grid(64), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [conv1d_1], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(reinterpret_tensor(buf20, (16, 1, 4), (4, 0, 1), 0), primals_21, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf21, (16, 1, 4), (4, 4, 1))
        buf22 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [w_3, mul_1, x_7], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_6.run(buf22, buf18, buf21, 1024, grid=grid(1024), stream=stream0)
    return (reinterpret_tensor(buf22, (4, 4, 4, 4, 4), (256, 64, 16, 4, 1), 0), primals_5, primals_7, primals_9, primals_10, primals_12, primals_14, primals_16, primals_17, primals_19, primals_21, reinterpret_tensor(buf0, (16, 6, 4, 4), (96, 16, 4, 1), 0), reinterpret_tensor(buf3, (16, 4, 4, 4), (64, 16, 4, 1), 0), buf5, buf7, buf9, reinterpret_tensor(buf11, (16, 1, 4), (4, 4, 1), 0), buf12, buf14, buf16, buf18, reinterpret_tensor(buf20, (16, 1, 4), (4, 4, 1), 0), buf21, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 4, 4, 4), (256, 64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
