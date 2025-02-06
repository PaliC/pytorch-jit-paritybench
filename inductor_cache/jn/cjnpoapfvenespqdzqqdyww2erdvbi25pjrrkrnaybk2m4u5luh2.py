# AOT ID: ['14_forward']
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
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %primals_3), kwargs = {})
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
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %sum_1), kwargs = {})
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
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %sum_2), kwargs = {})
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


# kernel path: inductor_cache/vm/cvmtyosu6wo5f2zrhqg5a5nyc7uojrxpqpdspew6aamknnq2bf6h.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => gt, mul_2, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_5, %div, %primals_4, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul_2), kwargs = {})
triton_poi_fused_convolution_leaky_relu_3 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_3(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xz/cxztredtxvestwyu2qe56nn6gstk4opktquntugvby5pzfdhzma3.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_1, %convolution_3, %convolution_5, %convolution_7], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 8, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 16*((-4) + x1) + 64*x2), tmp13 & xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-4) + x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 12, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr4 + (x0 + 16*((-8) + x1) + 64*x2), tmp22 & xmask, other=0.0)
    tmp24 = tl.load(in_ptr5 + ((-8) + x1), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tmp0 >= tmp20
    tmp29 = tl.full([1], 16, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr6 + (x0 + 16*((-12) + x1) + 64*x2), tmp28 & xmask, other=0.0)
    tmp32 = tl.load(in_ptr7 + ((-12) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp28, tmp33, tmp34)
    tmp36 = tl.where(tmp22, tmp27, tmp35)
    tmp37 = tl.where(tmp13, tmp18, tmp36)
    tmp38 = tl.where(tmp4, tmp9, tmp37)
    tl.store(out_ptr0 + (x3), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jk/cjkexmu62s5xdtebrtcxt5nv5fijpf72hqh5gv5clifjftbtxp5s.py
# Topologically Sorted Source Nodes: [mv_8], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_8 => mul_20, sum_17
# Graph fragment:
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %primals_36), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [1]), kwargs = {})
triton_per_fused_mv_5 = async_compile.triton('triton_per_fused_mv_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mv_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mv_5(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 144*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wo/cwoixokaltnx2woqrqzpin2hcpmd5glfm4in76jrhwvqx67nx7vh.py
# Topologically Sorted Source Nodes: [weight_8], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight_8 => div_8
# Graph fragment:
#   %div_8 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_34, %sum_18), kwargs = {})
triton_poi_fused_div_6 = async_compile.triton('triton_poi_fused_div_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
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


# kernel path: inductor_cache/na/cnarmirmy6f4qalui2kxslhdsdncmzt4fmwfdcazvwfgupduxxvy.py
# Topologically Sorted Source Nodes: [conv2d_8, out_1], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   conv2d_8 => convolution_8
#   out_1 => add
# Graph fragment:
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %div_8, %primals_37, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %primals_5), kwargs = {})
triton_poi_fused_add_convolution_7 = async_compile.triton('triton_poi_fused_add_convolution_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (36, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_6, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (36, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (36, ), (1, ))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (36, ), (1, ))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (36, ), (1, ))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (36, ), (1, ))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (36, ), (1, ))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (36, ), (1, ))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (144, ), (1, ))
    assert_size_stride(primals_37, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_1, primals_3, buf0, 4, 36, grid=grid(4), stream=stream0)
        buf1 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_2, buf0, buf1, 1, grid=grid(1), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_1, buf1, buf2, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_5, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 4, 4), (64, 16, 4, 1))
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_3.run(buf5, primals_4, buf4, 256, grid=grid(256), stream=stream0)
        del primals_4
        buf6 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mv_1], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_6, primals_8, buf6, 4, 36, grid=grid(4), stream=stream0)
        buf7 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_1], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_7, buf6, buf7, 1, grid=grid(1), stream=stream0)
        buf8 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_1], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_6, buf7, buf8, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf5, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 4, 4, 4), (64, 16, 4, 1))
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [mv_2], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_10, primals_12, buf10, 4, 36, grid=grid(4), stream=stream0)
        buf11 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_2], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_11, buf10, buf11, 1, grid=grid(1), stream=stream0)
        buf12 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_2], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_10, buf11, buf12, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(primals_5, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 4, 4, 4), (64, 16, 4, 1))
        buf14 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf15 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_3.run(buf15, primals_13, buf14, 256, grid=grid(256), stream=stream0)
        del primals_13
        buf16 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [mv_3], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_14, primals_16, buf16, 4, 36, grid=grid(4), stream=stream0)
        buf17 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_3], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_15, buf16, buf17, 1, grid=grid(1), stream=stream0)
        buf18 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_3], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_14, buf17, buf18, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf15, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 4, 4, 4), (64, 16, 4, 1))
        buf20 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [mv_4], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_18, primals_20, buf20, 4, 36, grid=grid(4), stream=stream0)
        buf21 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_4], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_19, buf20, buf21, 1, grid=grid(1), stream=stream0)
        buf22 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_4], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_18, buf21, buf22, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(primals_5, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 4, 4, 4), (64, 16, 4, 1))
        buf24 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf25 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_3.run(buf25, primals_21, buf24, 256, grid=grid(256), stream=stream0)
        del primals_21
        buf26 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [mv_5], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_22, primals_24, buf26, 4, 36, grid=grid(4), stream=stream0)
        buf27 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_5], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_23, buf26, buf27, 1, grid=grid(1), stream=stream0)
        buf28 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_5], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_22, buf27, buf28, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf25, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 4, 4, 4), (64, 16, 4, 1))
        buf30 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [mv_6], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_26, primals_28, buf30, 4, 36, grid=grid(4), stream=stream0)
        buf31 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_6], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_27, buf30, buf31, 1, grid=grid(1), stream=stream0)
        buf32 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_6], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_26, buf31, buf32, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(primals_5, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 4, 4, 4), (64, 16, 4, 1))
        buf34 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf35 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_3.run(buf35, primals_29, buf34, 256, grid=grid(256), stream=stream0)
        del primals_29
        buf36 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [mv_7], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_30, primals_32, buf36, 4, 36, grid=grid(4), stream=stream0)
        buf37 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_7], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_31, buf36, buf37, 1, grid=grid(1), stream=stream0)
        buf38 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_7], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_30, buf37, buf38, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf35, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 4, 4, 4), (64, 16, 4, 1))
        buf40 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf9, primals_9, buf19, primals_17, buf29, primals_25, buf39, primals_33, buf40, 1024, grid=grid(1024), stream=stream0)
        del buf19
        del buf29
        del buf39
        del buf9
        del primals_17
        del primals_25
        del primals_33
        del primals_9
        buf41 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [mv_8], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_5.run(primals_34, primals_36, buf41, 4, 144, grid=grid(4), stream=stream0)
        buf42 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_8], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_1.run(primals_35, buf41, buf42, 1, grid=grid(1), stream=stream0)
        del buf41
        buf43 = empty_strided_cuda((4, 16, 3, 3), (144, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_8], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_6.run(primals_34, buf42, buf43, 576, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf40, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 4, 4, 4), (64, 16, 4, 1))
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [conv2d_8, out_1], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_7.run(buf45, primals_37, primals_5, 256, grid=grid(256), stream=stream0)
        del primals_37
    return (buf45, buf2, buf8, buf12, buf18, buf22, buf28, buf32, buf38, buf43, primals_1, primals_2, primals_3, primals_5, primals_6, primals_7, primals_8, primals_10, primals_11, primals_12, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_30, primals_31, primals_32, primals_34, primals_35, primals_36, buf1, buf2, buf4, buf5, buf7, buf8, buf11, buf12, buf14, buf15, buf17, buf18, buf21, buf22, buf24, buf25, buf27, buf28, buf31, buf32, buf34, buf35, buf37, buf38, buf40, buf42, buf43, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
