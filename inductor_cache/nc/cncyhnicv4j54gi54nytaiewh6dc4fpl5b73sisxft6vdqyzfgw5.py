# AOT ID: ['41_forward']
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


# kernel path: inductor_cache/li/cliqf6vwvkgqzazlb2utq2g46t7jhkkr2dedr6txkfoore2v5vmh.py
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
    size_hints={'x': 128, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mv_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mv_0(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 15*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5a/c5aavtfwwako7mfiicuwafwkvqigrswcvi3xsf5ehodbz3hefqx4.py
# Topologically Sorted Source Nodes: [sigma], Original ATen: [aten.dot]
# Source node to ATen node mapping:
#   sigma => mul_1, sum_2
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %sum_1), kwargs = {})
#   %sum_2 : [num_users=3] = call_function[target=torch.ops.aten.sum.default](args = (%mul_1,), kwargs = {})
triton_per_fused_dot_1 = async_compile.triton('triton_per_fused_dot_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_dot_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_dot_1(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/km/ckmiglfz5slsyywrngw4ztn7sdrrikknvgp2w4iatkkftga37bi5.py
# Topologically Sorted Source Nodes: [weight, weight_8], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight => div
#   weight_8 => div_8
# Graph fragment:
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %sum_2), kwargs = {})
#   %div_8 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %sum_2), kwargs = {})
triton_poi_fused_div_2 = async_compile.triton('triton_poi_fused_div_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ot/cotvx3ncvtdyv7xl4x7t6v6ojrtwyw2jbb2j5o4yzcgccvz7tra7.py
# Topologically Sorted Source Nodes: [x, x_1, x_16, x_17], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => gt, mul_2, where
#   x_16 => convolution_8
#   x_17 => gt_7, mul_25, where_7
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_5, %div, %primals_4, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul_2), kwargs = {})
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_34, %div_8, %primals_4, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt_7 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_8, 0), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_8, 0.1), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %convolution_8, %mul_25), kwargs = {})
triton_poi_fused_convolution_leaky_relu_3 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_3(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/25/c25dx6byy3f4d2rm3bcg4w5ordv6gjmo4tsid2jjhust7rhhpz2b.py
# Topologically Sorted Source Nodes: [mv_1], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_1 => mul_3, sum_3
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %primals_8), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [1]), kwargs = {})
triton_red_fused_mv_4 = async_compile.triton('triton_red_fused_mv_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_4(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 1312*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w5/cw577avgdhhkof47qddb3d4dyc4sqtlojkgzmurbwkm5exjf4dck.py
# Topologically Sorted Source Nodes: [weight_1, weight_9], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight_1 => div_1
#   weight_9 => div_9
# Graph fragment:
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_6, %sum_4), kwargs = {})
#   %div_9 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_6, %sum_4), kwargs = {})
triton_poi_fused_div_5 = async_compile.triton('triton_poi_fused_div_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 167936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr1 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/j2/cj2kvrctpbua5xairoifg5cykk6ktvibqg24w2sk6bkld544hkrh.py
# Topologically Sorted Source Nodes: [x_2, x_3, x_18, x_19], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_18 => convolution_9
#   x_19 => gt_8, mul_28, where_8
#   x_2 => convolution_1
#   x_3 => gt_1, mul_5, where_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where, %div_1, %primals_9, [2], [20], [1], False, [0], 4), kwargs = {})
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.1), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_1, %mul_5), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_7, %div_9, %primals_9, [2], [20], [1], False, [0], 4), kwargs = {})
#   %gt_8 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_9, 0), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_9, 0.1), kwargs = {})
#   %where_8 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %convolution_9, %mul_28), kwargs = {})
triton_poi_fused_convolution_leaky_relu_6 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_6(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 32) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/w3/cw3ujhrahohfur4dsvsll4gpt27f4im5scvksfgsrl3uk67fus5e.py
# Topologically Sorted Source Nodes: [mv_2], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_2 => mul_6, sum_5
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %primals_12), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_6, [1]), kwargs = {})
triton_per_fused_mv_7 = async_compile.triton('triton_per_fused_mv_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mv_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mv_7(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 328
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 328*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/er/cerrlcukjyuqoledp5yab7l47kqk5nbvcrol4vmqj5emqnlup4xw.py
# Topologically Sorted Source Nodes: [sigma_2], Original ATen: [aten.dot]
# Source node to ATen node mapping:
#   sigma_2 => mul_7, sum_6
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_11, %sum_5), kwargs = {})
#   %sum_6 : [num_users=3] = call_function[target=torch.ops.aten.sum.default](args = (%mul_7,), kwargs = {})
triton_per_fused_dot_8 = async_compile.triton('triton_per_fused_dot_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_dot_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_dot_8(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/yz/cyzcmrt4gzztkxve3d7uibd3i3ng44eqw7ld573rmrc3doaqo3wr.py
# Topologically Sorted Source Nodes: [weight_2, weight_10], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight_10 => div_10
#   weight_2 => div_2
# Graph fragment:
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_10, %sum_6), kwargs = {})
#   %div_10 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_10, %sum_6), kwargs = {})
triton_poi_fused_div_9 = async_compile.triton('triton_poi_fused_div_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_9(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 83968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cu/ccukg37ucaroiavjyjbnwxjvs5pse2wvptrb3l7ro24uqppqqm4h.py
# Topologically Sorted Source Nodes: [x_4, x_5, x_20, x_21], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_20 => convolution_10
#   x_21 => gt_9, mul_31, where_9
#   x_4 => convolution_2
#   x_5 => gt_2, mul_8, where_2
# Graph fragment:
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_1, %div_2, %primals_13, [2], [20], [1], False, [0], 16), kwargs = {})
#   %gt_2 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, 0.1), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convolution_2, %mul_8), kwargs = {})
#   %convolution_10 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_8, %div_10, %primals_13, [2], [20], [1], False, [0], 16), kwargs = {})
#   %gt_9 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_10, 0), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_10, 0.1), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %convolution_10, %mul_31), kwargs = {})
triton_poi_fused_convolution_leaky_relu_10 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_10(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/6c/c6cz7oebauqvziz7rklcasee5qsorpwecsdz7foz4cv7esxsrp6e.py
# Topologically Sorted Source Nodes: [mv_3], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_3 => mul_9, sum_7
# Graph fragment:
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %primals_16), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_9, [1]), kwargs = {})
triton_per_fused_mv_11 = async_compile.triton('triton_per_fused_mv_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mv_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mv_11(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 656
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 656*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/2n/c2nwdmofgitdjesmf5k5t735olfwqk757d7mjquculh4zlptwde4.py
# Topologically Sorted Source Nodes: [sigma_3], Original ATen: [aten.dot]
# Source node to ATen node mapping:
#   sigma_3 => mul_10, sum_8
# Graph fragment:
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_15, %sum_7), kwargs = {})
#   %sum_8 : [num_users=3] = call_function[target=torch.ops.aten.sum.default](args = (%mul_10,), kwargs = {})
triton_per_fused_dot_12 = async_compile.triton('triton_per_fused_dot_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_dot_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_dot_12(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/xm/cxmft3d53cqqp5ruxz7hcdx6jjnv65x5oekseplmq3qnwig6ebby.py
# Topologically Sorted Source Nodes: [weight_3, weight_11], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight_11 => div_11
#   weight_3 => div_3
# Graph fragment:
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_14, %sum_8), kwargs = {})
#   %div_11 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_14, %sum_8), kwargs = {})
triton_poi_fused_div_13 = async_compile.triton('triton_poi_fused_div_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_13(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 335872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr1 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/re/cre64pcgmkqbrvcb5z2aq7em6aeruezqu6knzy6er2fwxlgr7is5.py
# Topologically Sorted Source Nodes: [x_6, x_7, x_22, x_23], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_22 => convolution_11
#   x_23 => gt_10, mul_34, where_10
#   x_6 => convolution_3
#   x_7 => gt_3, mul_11, where_3
# Graph fragment:
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_2, %div_3, %primals_17, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_3 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_3, 0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_3, 0.1), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %convolution_3, %mul_11), kwargs = {})
#   %convolution_11 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_9, %div_11, %primals_17, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_10 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_11, 0), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_11, 0.1), kwargs = {})
#   %where_10 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_10, %convolution_11, %mul_34), kwargs = {})
triton_poi_fused_convolution_leaky_relu_14 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_14(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/h3/ch3crhnbcmnyshjboeeayk2xms3twsvy5hyrmzxlfs4dsyfnhl7v.py
# Topologically Sorted Source Nodes: [mv_4], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_4 => mul_12, sum_9
# Graph fragment:
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %primals_20), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12, [1]), kwargs = {})
triton_red_fused_mv_15 = async_compile.triton('triton_red_fused_mv_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_15(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 1312*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4j/c4jpty42tn4bougadttqc26toexkv3zqs2jvpzuoaha7lhcmkf7s.py
# Topologically Sorted Source Nodes: [sigma_4], Original ATen: [aten.dot]
# Source node to ATen node mapping:
#   sigma_4 => mul_13, sum_10
# Graph fragment:
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_19, %sum_9), kwargs = {})
#   %sum_10 : [num_users=3] = call_function[target=torch.ops.aten.sum.default](args = (%mul_13,), kwargs = {})
triton_per_fused_dot_16 = async_compile.triton('triton_per_fused_dot_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_dot_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_dot_16(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/i6/ci6pzquqsmvbrxrpfzypowfgsunwrqxzn7qsxmi4owqtholw64y6.py
# Topologically Sorted Source Nodes: [weight_4, weight_12], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight_12 => div_12
#   weight_4 => div_4
# Graph fragment:
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_18, %sum_10), kwargs = {})
#   %div_12 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_18, %sum_10), kwargs = {})
triton_poi_fused_div_17 = async_compile.triton('triton_poi_fused_div_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_17(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1343488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr1 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/ue/cuelp4r464e4pc7yhngcyp4qzgrp63lvf25yizojuywb5pcpz4gk.py
# Topologically Sorted Source Nodes: [x_8, x_9, x_24, x_25], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_24 => convolution_12
#   x_25 => gt_11, mul_37, where_11
#   x_8 => convolution_4
#   x_9 => gt_4, mul_14, where_4
# Graph fragment:
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_3, %div_4, %primals_21, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_4 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_4, 0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_4, 0.1), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %convolution_4, %mul_14), kwargs = {})
#   %convolution_12 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_10, %div_12, %primals_21, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_11 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_12, 0), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_12, 0.1), kwargs = {})
#   %where_11 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_11, %convolution_12, %mul_37), kwargs = {})
triton_poi_fused_convolution_leaky_relu_18 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_18(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x2), tmp4, None)
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr1 + (x2), tmp10, None)
    tl.store(in_out_ptr1 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/nz/cnztux6y24uapqtdm5d3yt4rurcbu2cxacbtngorhq7hvwypt3hx.py
# Topologically Sorted Source Nodes: [mv_5], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_5 => mul_15, sum_11
# Graph fragment:
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %primals_24), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [1]), kwargs = {})
triton_red_fused_mv_19 = async_compile.triton('triton_red_fused_mv_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_19(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2624*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sk/cskvcceiqkmgrxtnpzkaeohvjilj3tebjeqrogoecozlmnya3eem.py
# Topologically Sorted Source Nodes: [weight_5, weight_13], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight_13 => div_13
#   weight_5 => div_5
# Graph fragment:
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_22, %sum_12), kwargs = {})
#   %div_13 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_22, %sum_12), kwargs = {})
triton_poi_fused_div_20 = async_compile.triton('triton_poi_fused_div_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_20(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2686976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr1 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/vm/cvmq6nuolfdz7wrmaf57gkgtfdehqd4nw3w4kvqtcenvn5bkck4e.py
# Topologically Sorted Source Nodes: [mv_6], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_6 => mul_18, sum_13
# Graph fragment:
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %primals_28), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_18, [1]), kwargs = {})
triton_red_fused_mv_21 = async_compile.triton('triton_red_fused_mv_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_21(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 5120*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/og/coggvvmzvkalogpojbts6wjf3dhq24k7bxifwisapsilaohmrkpx.py
# Topologically Sorted Source Nodes: [weight_6, weight_14], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight_14 => div_14
#   weight_6 => div_6
# Graph fragment:
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_26, %sum_14), kwargs = {})
#   %div_14 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_26, %sum_14), kwargs = {})
triton_poi_fused_div_22 = async_compile.triton('triton_poi_fused_div_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_22(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr1 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/wb/cwb4zqwmqidnzm4u3gl5zyttcs4epqb5kq366pyn6vrdnjf4amtg.py
# Topologically Sorted Source Nodes: [mv_7, sigma_7, weight_7, weight_15], Original ATen: [aten.mv, aten.dot, aten.div]
# Source node to ATen node mapping:
#   mv_7 => mul_21, sum_15
#   sigma_7 => mul_22, sum_16
#   weight_15 => div_15
#   weight_7 => div_7
# Graph fragment:
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %primals_32), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_21, [1]), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_31, %sum_15), kwargs = {})
#   %sum_16 : [num_users=3] = call_function[target=torch.ops.aten.sum.default](args = (%mul_22,), kwargs = {})
#   %div_7 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_30, %sum_16), kwargs = {})
#   %div_15 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_30, %sum_16), kwargs = {})
triton_red_fused_div_dot_mv_23 = async_compile.triton('triton_red_fused_div_dot_mv_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_dot_mv_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_dot_mv_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, 1])
    tmp8 = tmp7 * tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp9 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tmp9 / tmp8
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp10, rmask)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp10, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/37/c37n2ezk5troqs7lxyz5uztrlh4qyked6fpg3kt7rualyridfdbo.py
# Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_14 => convolution_7
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_6, %div_7, %primals_33, [1], [1], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s7/cs74a6lybznbh6hoydc44uq5oa52xsdetxm3xb4ffqpxnnmavevl.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   y => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%unsqueeze, [1, 4], [1, 2], [0, 2]), kwargs = {})
triton_poi_fused_avg_pool2d_25 = async_compile.triton('triton_poi_fused_avg_pool2d_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 132
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 33)
    x1 = xindex // 33
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tmp1 & tmp3
    tmp5 = (-2) + 2*x0
    tmp6 = tmp5 >= tmp0
    tmp7 = tl.full([1], 64, tl.int64)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tmp4 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-2) + 2*x0 + 64*x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = (-1) + 2*x0
    tmp13 = tmp12 >= tmp0
    tmp14 = tmp12 < tmp7
    tmp15 = tmp13 & tmp14
    tmp16 = tmp4 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-1) + 2*x0 + 64*x1), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 2*x0
    tmp20 = tmp19 >= tmp0
    tmp21 = tmp19 < tmp7
    tmp22 = tmp20 & tmp21
    tmp23 = tmp4 & tmp22
    tmp24 = tl.load(in_ptr0 + (2*x0 + 64*x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 1 + 2*x0
    tmp27 = tmp26 >= tmp0
    tmp28 = tmp26 < tmp7
    tmp29 = tmp27 & tmp28
    tmp30 = tmp4 & tmp29
    tmp31 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = 2 + ((-2)*x0) + ((66) * ((66) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (66)))
    tmp34 = tmp32 / tmp33
    tl.store(out_ptr0 + (x2), tmp34, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jo/cjoyn6gnd44g2qpeopfmcj5ihra5uny4uovjo422ppa46s24ojn4.py
# Topologically Sorted Source Nodes: [_weight_norm, _weight_norm_8], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm => div_16, mul_46, pow_1, pow_2, sum_33
#   _weight_norm_8 => mul_61
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_36, 2), kwargs = {})
#   %sum_33 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_33, 0.5), kwargs = {})
#   %div_16 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_35, %pow_2), kwargs = {})
#   %mul_46 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_36, %div_16), kwargs = {})
#   %mul_61 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_36, %div_16), kwargs = {})
triton_per_fused__weight_norm_interface_26 = async_compile.triton('triton_per_fused__weight_norm_interface_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_26(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 15*x0), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 15*x0), tmp9, rmask & xmask)
    tl.store(out_ptr1 + (r1 + 15*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ve/cvep42wmwdpmaf6nq673zjmpwr7ip3dinfmgav5dfrngxzumkwno.py
# Topologically Sorted Source Nodes: [x_32, x_33, x_48, x_49], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_32 => convolution_16
#   x_33 => gt_14, mul_47, where_14
#   x_48 => convolution_24
#   x_49 => gt_21, mul_62, where_21
# Graph fragment:
#   %convolution_16 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%squeeze, %mul_46, %primals_37, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt_14 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_16, 0), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_16, 0.1), kwargs = {})
#   %where_14 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %convolution_16, %mul_47), kwargs = {})
#   %convolution_24 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%squeeze_1, %mul_61, %primals_37, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt_21 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_24, 0), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_24, 0.1), kwargs = {})
#   %where_21 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_21, %convolution_24, %mul_62), kwargs = {})
triton_poi_fused_convolution_leaky_relu_27 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_27', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_27(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 33) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p3/cp3wsx56yx2omqm6tdyy7cne625ix2dluvu7l2ayp375lqpparfm.py
# Topologically Sorted Source Nodes: [_weight_norm_1, _weight_norm_9], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_1 => div_17, mul_48, pow_3, pow_4, sum_34
#   _weight_norm_9 => mul_63
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_39, 2), kwargs = {})
#   %sum_34 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1, 2], True), kwargs = {})
#   %pow_4 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_34, 0.5), kwargs = {})
#   %div_17 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_38, %pow_4), kwargs = {})
#   %mul_48 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_39, %div_17), kwargs = {})
#   %mul_63 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_39, %div_17), kwargs = {})
triton_red_fused__weight_norm_interface_28 = async_compile.triton('triton_red_fused__weight_norm_interface_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_28(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 1312*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + 1312*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 / tmp5
        tmp9 = tmp6 * tmp8
        tl.store(out_ptr0 + (r1 + 1312*x0), tmp9, rmask & xmask)
        tl.store(out_ptr1 + (r1 + 1312*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ye/cyed5kr572vsbpty7n3kl4a42wp5i5exuth3ovgvabgkeyal5y7p.py
# Topologically Sorted Source Nodes: [x_34, x_35, x_50, x_51], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_34 => convolution_17
#   x_35 => gt_15, mul_49, where_15
#   x_50 => convolution_25
#   x_51 => gt_22, mul_64, where_22
# Graph fragment:
#   %convolution_17 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_14, %mul_48, %primals_40, [2], [20], [1], False, [0], 4), kwargs = {})
#   %gt_15 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_17, 0), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_17, 0.1), kwargs = {})
#   %where_15 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_15, %convolution_17, %mul_49), kwargs = {})
#   %convolution_25 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_21, %mul_63, %primals_40, [2], [20], [1], False, [0], 4), kwargs = {})
#   %gt_22 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_25, 0), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_25, 0.1), kwargs = {})
#   %where_22 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_22, %convolution_25, %mul_64), kwargs = {})
triton_poi_fused_convolution_leaky_relu_29 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_29(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 17) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uj/cujxbhfwyuxufqliie27jgd3mqkd7s75ddb3ezvebgp6q5smop3s.py
# Topologically Sorted Source Nodes: [_weight_norm_2, _weight_norm_10], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_10 => mul_65
#   _weight_norm_2 => div_18, mul_50, pow_5, pow_6, sum_35
# Graph fragment:
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_42, 2), kwargs = {})
#   %sum_35 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1, 2], True), kwargs = {})
#   %pow_6 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_35, 0.5), kwargs = {})
#   %div_18 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_41, %pow_6), kwargs = {})
#   %mul_50 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_42, %div_18), kwargs = {})
#   %mul_65 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_42, %div_18), kwargs = {})
triton_per_fused__weight_norm_interface_30 = async_compile.triton('triton_per_fused__weight_norm_interface_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_30(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 328
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 328*x0), rmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr0 + (r1 + 328*x0), tmp9, rmask)
    tl.store(out_ptr1 + (r1 + 328*x0), tmp9, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/ib/cibmkd5hymai7rt6u4ruf5rfexjwonp4vpv2zjqnm4nrybm5yviw.py
# Topologically Sorted Source Nodes: [x_36, x_37, x_52, x_53], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_36 => convolution_18
#   x_37 => gt_16, mul_51, where_16
#   x_52 => convolution_26
#   x_53 => gt_23, mul_66, where_23
# Graph fragment:
#   %convolution_18 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_15, %mul_50, %primals_43, [2], [20], [1], False, [0], 16), kwargs = {})
#   %gt_16 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_18, 0), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_18, 0.1), kwargs = {})
#   %where_16 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_16, %convolution_18, %mul_51), kwargs = {})
#   %convolution_26 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_22, %mul_65, %primals_43, [2], [20], [1], False, [0], 16), kwargs = {})
#   %gt_23 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_26, 0), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_26, 0.1), kwargs = {})
#   %where_23 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_23, %convolution_26, %mul_66), kwargs = {})
triton_poi_fused_convolution_leaky_relu_31 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_31(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 9) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fv/cfv5k2sy7otcvndcm5lijkk7hp5cec2j3fxbdhyxqxgdo4rmszyw.py
# Topologically Sorted Source Nodes: [_weight_norm_3, _weight_norm_11], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_11 => mul_67
#   _weight_norm_3 => div_19, mul_52, pow_7, pow_8, sum_36
# Graph fragment:
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_45, 2), kwargs = {})
#   %sum_36 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2], True), kwargs = {})
#   %pow_8 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_36, 0.5), kwargs = {})
#   %div_19 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_44, %pow_8), kwargs = {})
#   %mul_52 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_45, %div_19), kwargs = {})
#   %mul_67 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_45, %div_19), kwargs = {})
triton_per_fused__weight_norm_interface_32 = async_compile.triton('triton_per_fused__weight_norm_interface_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_32(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 656
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 656*x0), rmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr0 + (r1 + 656*x0), tmp9, rmask)
    tl.store(out_ptr1 + (r1 + 656*x0), tmp9, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/pt/cpt2mjraf7fztumxzjwjqprposvw5lgojxoij55xzp6aahpbefgy.py
# Topologically Sorted Source Nodes: [x_38, x_39, x_54, x_55], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_38 => convolution_19
#   x_39 => gt_17, mul_53, where_17
#   x_54 => convolution_27
#   x_55 => gt_24, mul_68, where_24
# Graph fragment:
#   %convolution_19 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_16, %mul_52, %primals_46, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_17 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_19, 0), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_19, 0.1), kwargs = {})
#   %where_17 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_17, %convolution_19, %mul_53), kwargs = {})
#   %convolution_27 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_23, %mul_67, %primals_46, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_24 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_27, 0), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_27, 0.1), kwargs = {})
#   %where_24 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_24, %convolution_27, %mul_68), kwargs = {})
triton_poi_fused_convolution_leaky_relu_33 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_33', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_33(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qu/cquxiwf4zkqvdg2nhrje5tuyjdjjmuk7jfw3plgi3xjp7yibv3ya.py
# Topologically Sorted Source Nodes: [_weight_norm_4, _weight_norm_12], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_12 => mul_69
#   _weight_norm_4 => div_20, mul_54, pow_10, pow_9, sum_37
# Graph fragment:
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_48, 2), kwargs = {})
#   %sum_37 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [1, 2], True), kwargs = {})
#   %pow_10 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_37, 0.5), kwargs = {})
#   %div_20 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_47, %pow_10), kwargs = {})
#   %mul_54 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_48, %div_20), kwargs = {})
#   %mul_69 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_48, %div_20), kwargs = {})
triton_red_fused__weight_norm_interface_34 = async_compile.triton('triton_red_fused__weight_norm_interface_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_34(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 1312*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + 1312*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 / tmp5
        tmp9 = tmp6 * tmp8
        tl.store(out_ptr0 + (r1 + 1312*x0), tmp9, rmask & xmask)
        tl.store(out_ptr1 + (r1 + 1312*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hv/chvqsydt4rcwkdjyj4jvbsk26q7vbwoyfwwdtyithqahdnzsrmx3.py
# Topologically Sorted Source Nodes: [_weight_norm_5, _weight_norm_13], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_13 => mul_71
#   _weight_norm_5 => div_21, mul_56, pow_11, pow_12, sum_38
# Graph fragment:
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_51, 2), kwargs = {})
#   %sum_38 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1, 2], True), kwargs = {})
#   %pow_12 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_38, 0.5), kwargs = {})
#   %div_21 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_50, %pow_12), kwargs = {})
#   %mul_56 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_51, %div_21), kwargs = {})
#   %mul_71 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_51, %div_21), kwargs = {})
triton_red_fused__weight_norm_interface_35 = async_compile.triton('triton_red_fused__weight_norm_interface_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_35(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2624*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + 2624*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 / tmp5
        tmp9 = tmp6 * tmp8
        tl.store(out_ptr0 + (r1 + 2624*x0), tmp9, rmask & xmask)
        tl.store(out_ptr1 + (r1 + 2624*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/66/c665tsyetkkbcwdk7yajve4qw6fr3y26wjn4pfrzmbyt5o6ctsfs.py
# Topologically Sorted Source Nodes: [_weight_norm_6, _weight_norm_14], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_14 => mul_73
#   _weight_norm_6 => div_22, mul_58, pow_13, pow_14, sum_39
# Graph fragment:
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_54, 2), kwargs = {})
#   %sum_39 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_13, [1, 2], True), kwargs = {})
#   %pow_14 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_39, 0.5), kwargs = {})
#   %div_22 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_53, %pow_14), kwargs = {})
#   %mul_58 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_54, %div_22), kwargs = {})
#   %mul_73 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_54, %div_22), kwargs = {})
triton_red_fused__weight_norm_interface_36 = async_compile.triton('triton_red_fused__weight_norm_interface_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_36(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 5120*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + 5120*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 / tmp5
        tmp9 = tmp6 * tmp8
        tl.store(out_ptr0 + (r1 + 5120*x0), tmp9, rmask & xmask)
        tl.store(out_ptr1 + (r1 + 5120*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yl/cylo52rgr3mol5xtubm4p7jc7xfbddvx57eh6bu7fkucyxwv42lb.py
# Topologically Sorted Source Nodes: [_weight_norm_7, _weight_norm_15], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_15 => mul_75
#   _weight_norm_7 => div_23, mul_60, pow_15, pow_16, sum_40
# Graph fragment:
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_57, 2), kwargs = {})
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_15, [1, 2], True), kwargs = {})
#   %pow_16 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_40, 0.5), kwargs = {})
#   %div_23 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_56, %pow_16), kwargs = {})
#   %mul_60 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_57, %div_23), kwargs = {})
#   %mul_75 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_57, %div_23), kwargs = {})
triton_red_fused__weight_norm_interface_37 = async_compile.triton('triton_red_fused__weight_norm_interface_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_37(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp6 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp8 / tmp5
        tmp10 = tmp6 * tmp9
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp10, rmask)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp10, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/t3/ct3njwroxhodg77lvjdyelpblyanje37llcnsd4ozdp6lxh2bay6.py
# Topologically Sorted Source Nodes: [y_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   y_1 => avg_pool2d_2
# Graph fragment:
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%unsqueeze_2, [1, 4], [1, 2], [0, 2]), kwargs = {})
triton_poi_fused_avg_pool2d_38 = async_compile.triton('triton_poi_fused_avg_pool2d_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 68
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 17)
    x1 = xindex // 17
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tmp1 & tmp3
    tmp5 = (-2) + 2*x0
    tmp6 = tmp5 >= tmp0
    tmp7 = tl.full([1], 33, tl.int64)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tmp4 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-2) + 2*x0 + 33*x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = (-1) + 2*x0
    tmp13 = tmp12 >= tmp0
    tmp14 = tmp12 < tmp7
    tmp15 = tmp13 & tmp14
    tmp16 = tmp4 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-1) + 2*x0 + 33*x1), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 2*x0
    tmp20 = tmp19 >= tmp0
    tmp21 = tmp19 < tmp7
    tmp22 = tmp20 & tmp21
    tmp23 = tmp4 & tmp22
    tmp24 = tl.load(in_ptr0 + (2*x0 + 33*x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 1 + 2*x0
    tmp27 = tmp26 >= tmp0
    tmp28 = tmp26 < tmp7
    tmp29 = tmp27 & tmp28
    tmp30 = tmp4 & tmp29
    tmp31 = tl.load(in_ptr0 + (1 + 2*x0 + 33*x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = 2 + ((-2)*x0) + ((35) * ((35) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (35)))
    tmp34 = tmp32 / tmp33
    tl.store(out_ptr0 + (x2), tmp34, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j7/cj7su575ykboczxd6lap5wkt73jmqwp7oba6pfxnzguu5t4mszh7.py
# Topologically Sorted Source Nodes: [x_66, x_67, x_82, x_83], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_66 => convolution_33
#   x_67 => gt_29, mul_79, where_29
#   x_82 => convolution_41
#   x_83 => gt_36, mul_94, where_36
# Graph fragment:
#   %convolution_33 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_28, %mul_78, %primals_64, [2], [20], [1], False, [0], 4), kwargs = {})
#   %gt_29 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_33, 0), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_33, 0.1), kwargs = {})
#   %where_29 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_29, %convolution_33, %mul_79), kwargs = {})
#   %convolution_41 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_35, %mul_93, %primals_64, [2], [20], [1], False, [0], 4), kwargs = {})
#   %gt_36 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_41, 0), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_41, 0.1), kwargs = {})
#   %where_36 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_36, %convolution_41, %mul_94), kwargs = {})
triton_poi_fused_convolution_leaky_relu_39 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_39', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_39(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 9) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4t/c4tnvdpyhf2m7v2evzjkj4fqw6w4g4guyleljbjkd5cricfyxp46.py
# Topologically Sorted Source Nodes: [x_68, x_69, x_84, x_85], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_68 => convolution_34
#   x_69 => gt_30, mul_81, where_30
#   x_84 => convolution_42
#   x_85 => gt_37, mul_96, where_37
# Graph fragment:
#   %convolution_34 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_29, %mul_80, %primals_67, [2], [20], [1], False, [0], 16), kwargs = {})
#   %gt_30 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_34, 0), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_34, 0.1), kwargs = {})
#   %where_30 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_30, %convolution_34, %mul_81), kwargs = {})
#   %convolution_42 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_36, %mul_95, %primals_67, [2], [20], [1], False, [0], 16), kwargs = {})
#   %gt_37 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_42, 0), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_42, 0.1), kwargs = {})
#   %where_37 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_37, %convolution_42, %mul_96), kwargs = {})
triton_poi_fused_convolution_leaky_relu_40 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_40', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_40(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 5) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sf/csfdqbg7uncu4qiclwfop4x7d7ni3aue5fsopcff6gakohetcauq.py
# Topologically Sorted Source Nodes: [x_70, x_71, x_86, x_87], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_70 => convolution_35
#   x_71 => gt_31, mul_83, where_31
#   x_86 => convolution_43
#   x_87 => gt_38, mul_98, where_38
# Graph fragment:
#   %convolution_35 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_30, %mul_82, %primals_70, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_31 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_35, 0), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_35, 0.1), kwargs = {})
#   %where_31 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_31, %convolution_35, %mul_83), kwargs = {})
#   %convolution_43 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_37, %mul_97, %primals_70, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_38 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_43, 0), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_43, 0.1), kwargs = {})
#   %where_38 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_38, %convolution_43, %mul_98), kwargs = {})
triton_poi_fused_convolution_leaky_relu_41 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_41', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_41(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 2) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82 = args
    args.clear()
    assert_size_stride(primals_1, (128, 1, 15), (15, 15, 1))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (15, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (4, 1, 64), (64, 64, 1))
    assert_size_stride(primals_6, (128, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (1312, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (256, 8, 41), (328, 41, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (328, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (512, 16, 41), (656, 41, 1))
    assert_size_stride(primals_15, (512, ), (1, ))
    assert_size_stride(primals_16, (656, ), (1, ))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (1024, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_19, (1024, ), (1, ))
    assert_size_stride(primals_20, (1312, ), (1, ))
    assert_size_stride(primals_21, (1024, ), (1, ))
    assert_size_stride(primals_22, (1024, 64, 41), (2624, 41, 1))
    assert_size_stride(primals_23, (1024, ), (1, ))
    assert_size_stride(primals_24, (2624, ), (1, ))
    assert_size_stride(primals_25, (1024, ), (1, ))
    assert_size_stride(primals_26, (1024, 1024, 5), (5120, 5, 1))
    assert_size_stride(primals_27, (1024, ), (1, ))
    assert_size_stride(primals_28, (5120, ), (1, ))
    assert_size_stride(primals_29, (1024, ), (1, ))
    assert_size_stride(primals_30, (1, 1024, 3), (3072, 3, 1))
    assert_size_stride(primals_31, (1, ), (1, ))
    assert_size_stride(primals_32, (3072, ), (1, ))
    assert_size_stride(primals_33, (1, ), (1, ))
    assert_size_stride(primals_34, (4, 1, 64), (64, 64, 1))
    assert_size_stride(primals_35, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_36, (128, 1, 15), (15, 15, 1))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_39, (128, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_42, (256, 8, 41), (328, 41, 1))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_45, (512, 16, 41), (656, 41, 1))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_47, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_48, (1024, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_50, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_51, (1024, 64, 41), (2624, 41, 1))
    assert_size_stride(primals_52, (1024, ), (1, ))
    assert_size_stride(primals_53, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_54, (1024, 1024, 5), (5120, 5, 1))
    assert_size_stride(primals_55, (1024, ), (1, ))
    assert_size_stride(primals_56, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_57, (1, 1024, 3), (3072, 3, 1))
    assert_size_stride(primals_58, (1, ), (1, ))
    assert_size_stride(primals_59, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_60, (128, 1, 15), (15, 15, 1))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_63, (128, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_66, (256, 8, 41), (328, 41, 1))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_69, (512, 16, 41), (656, 41, 1))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_72, (1024, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_73, (1024, ), (1, ))
    assert_size_stride(primals_74, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_75, (1024, 64, 41), (2624, 41, 1))
    assert_size_stride(primals_76, (1024, ), (1, ))
    assert_size_stride(primals_77, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_78, (1024, 1024, 5), (5120, 5, 1))
    assert_size_stride(primals_79, (1024, ), (1, ))
    assert_size_stride(primals_80, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_81, (1, 1024, 3), (3072, 3, 1))
    assert_size_stride(primals_82, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_0.run(primals_1, primals_3, buf0, 128, 15, grid=grid(128), stream=stream0)
        buf1 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_dot_1.run(primals_2, buf0, buf1, 1, 128, grid=grid(1), stream=stream0)
        buf2 = empty_strided_cuda((128, 1, 15), (15, 15, 1), torch.float32)
        buf47 = empty_strided_cuda((128, 1, 15), (15, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight, weight_8], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_2.run(primals_1, buf1, buf2, buf47, 1920, grid=grid(1920), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_5, buf2, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf3, (4, 128, 64), (8192, 64, 1))
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(primals_34, buf47, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf48, (4, 128, 64), (8192, 64, 1))
        buf4 = empty_strided_cuda((4, 128, 64), (8192, 64, 1), torch.bool)
        buf5 = buf3; del buf3  # reuse
        buf49 = empty_strided_cuda((4, 128, 64), (8192, 64, 1), torch.bool)
        buf50 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_16, x_17], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_3.run(buf5, buf50, primals_4, buf4, buf49, 32768, grid=grid(32768), stream=stream0)
        del primals_4
        buf6 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mv_1], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_4.run(primals_6, primals_8, buf6, 128, 1312, grid=grid(128), stream=stream0)
        buf7 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_1], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_dot_1.run(primals_7, buf6, buf7, 1, 128, grid=grid(1), stream=stream0)
        buf8 = empty_strided_cuda((128, 32, 41), (1312, 41, 1), torch.float32)
        buf51 = empty_strided_cuda((128, 32, 41), (1312, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_1, weight_9], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(primals_6, buf7, buf8, buf51, 167936, grid=grid(167936), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf5, buf8, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf9, (4, 128, 32), (4096, 32, 1))
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf50, buf51, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf52, (4, 128, 32), (4096, 32, 1))
        buf10 = empty_strided_cuda((4, 128, 32), (4096, 32, 1), torch.bool)
        buf11 = buf9; del buf9  # reuse
        buf53 = empty_strided_cuda((4, 128, 32), (4096, 32, 1), torch.bool)
        buf54 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3, x_18, x_19], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_6.run(buf11, buf54, primals_9, buf10, buf53, 16384, grid=grid(16384), stream=stream0)
        del primals_9
        buf12 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv_2], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_7.run(primals_10, primals_12, buf12, 256, 328, grid=grid(256), stream=stream0)
        buf13 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_2], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_dot_8.run(primals_11, buf12, buf13, 1, 256, grid=grid(1), stream=stream0)
        buf14 = empty_strided_cuda((256, 8, 41), (328, 41, 1), torch.float32)
        buf55 = empty_strided_cuda((256, 8, 41), (328, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_2, weight_10], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_9.run(primals_10, buf13, buf14, buf55, 83968, grid=grid(83968), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf11, buf14, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf15, (4, 256, 16), (4096, 16, 1))
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf54, buf55, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf56, (4, 256, 16), (4096, 16, 1))
        buf16 = empty_strided_cuda((4, 256, 16), (4096, 16, 1), torch.bool)
        buf17 = buf15; del buf15  # reuse
        buf57 = empty_strided_cuda((4, 256, 16), (4096, 16, 1), torch.bool)
        buf58 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_4, x_5, x_20, x_21], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_10.run(buf17, buf58, primals_13, buf16, buf57, 16384, grid=grid(16384), stream=stream0)
        del primals_13
        buf18 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv_3], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_11.run(primals_14, primals_16, buf18, 512, 656, grid=grid(512), stream=stream0)
        buf19 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_3], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_dot_12.run(primals_15, buf18, buf19, 1, 512, grid=grid(1), stream=stream0)
        buf20 = empty_strided_cuda((512, 16, 41), (656, 41, 1), torch.float32)
        buf59 = empty_strided_cuda((512, 16, 41), (656, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_3, weight_11], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_13.run(primals_14, buf19, buf20, buf59, 335872, grid=grid(335872), stream=stream0)
        del primals_14
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf17, buf20, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf21, (4, 512, 4), (2048, 4, 1))
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf58, buf59, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf60, (4, 512, 4), (2048, 4, 1))
        buf22 = empty_strided_cuda((4, 512, 4), (2048, 4, 1), torch.bool)
        buf23 = buf21; del buf21  # reuse
        buf61 = empty_strided_cuda((4, 512, 4), (2048, 4, 1), torch.bool)
        buf62 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7, x_22, x_23], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_14.run(buf23, buf62, primals_17, buf22, buf61, 8192, grid=grid(8192), stream=stream0)
        del primals_17
        buf24 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv_4], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_15.run(primals_18, primals_20, buf24, 1024, 1312, grid=grid(1024), stream=stream0)
        buf25 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_4], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_dot_16.run(primals_19, buf24, buf25, 1, 1024, grid=grid(1), stream=stream0)
        buf26 = empty_strided_cuda((1024, 32, 41), (1312, 41, 1), torch.float32)
        buf63 = empty_strided_cuda((1024, 32, 41), (1312, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_4, weight_12], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_17.run(primals_18, buf25, buf26, buf63, 1343488, grid=grid(1343488), stream=stream0)
        del primals_18
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf23, buf26, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf27, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf62, buf63, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf64, (4, 1024, 1), (1024, 1, 1))
        buf28 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf29 = buf27; del buf27  # reuse
        buf65 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf66 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_8, x_9, x_24, x_25], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf29, buf66, primals_21, buf28, buf65, 4096, grid=grid(4096), stream=stream0)
        del primals_21
        buf30 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [mv_5], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_19.run(primals_22, primals_24, buf30, 1024, 2624, grid=grid(1024), stream=stream0)
        buf31 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_5], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_dot_16.run(primals_23, buf30, buf31, 1, 1024, grid=grid(1), stream=stream0)
        buf32 = empty_strided_cuda((1024, 64, 41), (2624, 41, 1), torch.float32)
        buf67 = empty_strided_cuda((1024, 64, 41), (2624, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_5, weight_13], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_20.run(primals_22, buf31, buf32, buf67, 2686976, grid=grid(2686976), stream=stream0)
        del primals_22
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf29, buf32, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf33, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf66, buf67, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf68, (4, 1024, 1), (1024, 1, 1))
        buf34 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf35 = buf33; del buf33  # reuse
        buf69 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf70 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_10, x_11, x_26, x_27], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf35, buf70, primals_25, buf34, buf69, 4096, grid=grid(4096), stream=stream0)
        del primals_25
        buf36 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [mv_6], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_21.run(primals_26, primals_28, buf36, 1024, 5120, grid=grid(1024), stream=stream0)
        buf37 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sigma_6], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_dot_16.run(primals_27, buf36, buf37, 1, 1024, grid=grid(1), stream=stream0)
        buf38 = empty_strided_cuda((1024, 1024, 5), (5120, 5, 1), torch.float32)
        buf71 = empty_strided_cuda((1024, 1024, 5), (5120, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_6, weight_14], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_22.run(primals_26, buf37, buf38, buf71, 5242880, grid=grid(5242880), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf35, buf38, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf39, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf70, buf71, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf72, (4, 1024, 1), (1024, 1, 1))
        buf40 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf41 = buf39; del buf39  # reuse
        buf73 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf74 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_12, x_13, x_28, x_29], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf41, buf74, primals_29, buf40, buf73, 4096, grid=grid(4096), stream=stream0)
        del primals_29
        buf42 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf43 = reinterpret_tensor(buf42, (), (), 0); del buf42  # reuse
        buf44 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        buf75 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mv_7, sigma_7, weight_7, weight_15], Original ATen: [aten.mv, aten.dot, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_dot_mv_23.run(buf43, primals_30, primals_32, primals_31, buf44, buf75, 1, 3072, grid=grid(1), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf41, buf44, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf45, (4, 1, 1), (1, 1, 1))
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf46, primals_33, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf74, buf75, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf76, (4, 1, 1), (1, 1, 1))
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf77, primals_33, 4, grid=grid(4), stream=stream0)
        del primals_33
        buf78 = empty_strided_cuda((4, 1, 1, 33), (33, 33, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_25.run(primals_5, buf78, 132, grid=grid(132), stream=stream0)
        buf79 = empty_strided_cuda((4, 1, 1, 33), (33, 33, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_hat], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_25.run(primals_34, buf79, 132, grid=grid(132), stream=stream0)
        buf80 = reinterpret_tensor(buf6, (128, 1, 1), (1, 128, 128), 0); del buf6  # reuse
        buf81 = reinterpret_tensor(buf80, (128, 1, 1), (1, 1, 1), 0); del buf80  # reuse
        buf82 = empty_strided_cuda((128, 1, 15), (15, 15, 1), torch.float32)
        buf127 = empty_strided_cuda((128, 1, 15), (15, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm, _weight_norm_8], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_26.run(buf81, primals_36, primals_35, buf82, buf127, 128, 15, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(reinterpret_tensor(buf78, (4, 1, 33), (33, 33, 1), 0), buf82, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf83, (4, 128, 33), (4224, 33, 1))
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(reinterpret_tensor(buf79, (4, 1, 33), (33, 33, 1), 0), buf127, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf128, (4, 128, 33), (4224, 33, 1))
        buf84 = empty_strided_cuda((4, 128, 33), (4224, 33, 1), torch.bool)
        buf85 = buf83; del buf83  # reuse
        buf129 = empty_strided_cuda((4, 128, 33), (4224, 33, 1), torch.bool)
        buf130 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_32, x_33, x_48, x_49], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_27.run(buf85, buf130, primals_37, buf84, buf129, 16896, grid=grid(16896), stream=stream0)
        del primals_37
        buf86 = empty_strided_cuda((128, 1, 1), (1, 128, 128), torch.float32)
        buf87 = reinterpret_tensor(buf86, (128, 1, 1), (1, 1, 1), 0); del buf86  # reuse
        buf88 = empty_strided_cuda((128, 32, 41), (1312, 41, 1), torch.float32)
        buf131 = empty_strided_cuda((128, 32, 41), (1312, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_1, _weight_norm_9], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_28.run(buf87, primals_39, primals_38, buf88, buf131, 128, 1312, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf85, buf88, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf89, (4, 128, 17), (2176, 17, 1))
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf130, buf131, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf132, (4, 128, 17), (2176, 17, 1))
        buf90 = empty_strided_cuda((4, 128, 17), (2176, 17, 1), torch.bool)
        buf91 = buf89; del buf89  # reuse
        buf133 = empty_strided_cuda((4, 128, 17), (2176, 17, 1), torch.bool)
        buf134 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_34, x_35, x_50, x_51], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_29.run(buf91, buf134, primals_40, buf90, buf133, 8704, grid=grid(8704), stream=stream0)
        del primals_40
        buf92 = reinterpret_tensor(buf12, (256, 1, 1), (1, 256, 256), 0); del buf12  # reuse
        buf93 = reinterpret_tensor(buf92, (256, 1, 1), (1, 1, 1), 0); del buf92  # reuse
        buf94 = empty_strided_cuda((256, 8, 41), (328, 41, 1), torch.float32)
        buf135 = empty_strided_cuda((256, 8, 41), (328, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_2, _weight_norm_10], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_30.run(buf93, primals_42, primals_41, buf94, buf135, 256, 328, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf91, buf94, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf95, (4, 256, 9), (2304, 9, 1))
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf134, buf135, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf136, (4, 256, 9), (2304, 9, 1))
        buf96 = empty_strided_cuda((4, 256, 9), (2304, 9, 1), torch.bool)
        buf97 = buf95; del buf95  # reuse
        buf137 = empty_strided_cuda((4, 256, 9), (2304, 9, 1), torch.bool)
        buf138 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_36, x_37, x_52, x_53], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_31.run(buf97, buf138, primals_43, buf96, buf137, 9216, grid=grid(9216), stream=stream0)
        del primals_43
        buf98 = reinterpret_tensor(buf18, (512, 1, 1), (1, 512, 512), 0); del buf18  # reuse
        buf99 = reinterpret_tensor(buf98, (512, 1, 1), (1, 1, 1), 0); del buf98  # reuse
        buf100 = empty_strided_cuda((512, 16, 41), (656, 41, 1), torch.float32)
        buf139 = empty_strided_cuda((512, 16, 41), (656, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_3, _weight_norm_11], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_32.run(buf99, primals_45, primals_44, buf100, buf139, 512, 656, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf97, buf100, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf101, (4, 512, 3), (1536, 3, 1))
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf138, buf139, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf140, (4, 512, 3), (1536, 3, 1))
        buf102 = empty_strided_cuda((4, 512, 3), (1536, 3, 1), torch.bool)
        buf103 = buf101; del buf101  # reuse
        buf141 = empty_strided_cuda((4, 512, 3), (1536, 3, 1), torch.bool)
        buf142 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_38, x_39, x_54, x_55], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_33.run(buf103, buf142, primals_46, buf102, buf141, 6144, grid=grid(6144), stream=stream0)
        del primals_46
        buf104 = reinterpret_tensor(buf36, (1024, 1, 1), (1, 1024, 1024), 0); del buf36  # reuse
        buf105 = reinterpret_tensor(buf104, (1024, 1, 1), (1, 1, 1), 0); del buf104  # reuse
        buf106 = empty_strided_cuda((1024, 32, 41), (1312, 41, 1), torch.float32)
        buf143 = empty_strided_cuda((1024, 32, 41), (1312, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_4, _weight_norm_12], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_34.run(buf105, primals_48, primals_47, buf106, buf143, 1024, 1312, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf103, buf106, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf107, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf142, buf143, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf144, (4, 1024, 1), (1024, 1, 1))
        buf108 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf109 = buf107; del buf107  # reuse
        buf145 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf146 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_40, x_41, x_56, x_57], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf109, buf146, primals_49, buf108, buf145, 4096, grid=grid(4096), stream=stream0)
        del primals_49
        buf110 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf111 = reinterpret_tensor(buf110, (1024, 1, 1), (1, 1, 1), 0); del buf110  # reuse
        buf112 = empty_strided_cuda((1024, 64, 41), (2624, 41, 1), torch.float32)
        buf147 = empty_strided_cuda((1024, 64, 41), (2624, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_5, _weight_norm_13], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_35.run(buf111, primals_51, primals_50, buf112, buf147, 1024, 2624, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf109, buf112, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf113, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf146, buf147, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf148, (4, 1024, 1), (1024, 1, 1))
        buf114 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf115 = buf113; del buf113  # reuse
        buf149 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf150 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_43, x_58, x_59], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf115, buf150, primals_52, buf114, buf149, 4096, grid=grid(4096), stream=stream0)
        del primals_52
        buf116 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf117 = reinterpret_tensor(buf116, (1024, 1, 1), (1, 1, 1), 0); del buf116  # reuse
        buf118 = empty_strided_cuda((1024, 1024, 5), (5120, 5, 1), torch.float32)
        buf151 = empty_strided_cuda((1024, 1024, 5), (5120, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_6, _weight_norm_14], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_36.run(buf117, primals_54, primals_53, buf118, buf151, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf115, buf118, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf119, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf150, buf151, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf152, (4, 1024, 1), (1024, 1, 1))
        buf120 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf121 = buf119; del buf119  # reuse
        buf153 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf154 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_44, x_45, x_60, x_61], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf121, buf154, primals_55, buf120, buf153, 4096, grid=grid(4096), stream=stream0)
        del primals_55
        buf122 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf123 = buf122; del buf122  # reuse
        buf124 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        buf155 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_7, _weight_norm_15], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_37.run(buf123, primals_57, primals_56, buf124, buf155, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf121, buf124, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf125, (4, 1, 1), (1, 1, 1))
        buf126 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf126, primals_58, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf154, buf155, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf156, (4, 1, 1), (1, 1, 1))
        buf157 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf157, primals_58, 4, grid=grid(4), stream=stream0)
        del primals_58
        buf158 = empty_strided_cuda((4, 1, 1, 17), (17, 17, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_1], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf78, buf158, 68, grid=grid(68), stream=stream0)
        buf159 = empty_strided_cuda((4, 1, 1, 17), (17, 17, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_hat_1], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_38.run(buf79, buf159, 68, grid=grid(68), stream=stream0)
        buf160 = empty_strided_cuda((128, 1, 1), (1, 128, 128), torch.float32)
        buf161 = reinterpret_tensor(buf160, (128, 1, 1), (1, 1, 1), 0); del buf160  # reuse
        buf162 = empty_strided_cuda((128, 1, 15), (15, 15, 1), torch.float32)
        buf207 = empty_strided_cuda((128, 1, 15), (15, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_16, _weight_norm_24], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_26.run(buf161, primals_60, primals_59, buf162, buf207, 128, 15, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(reinterpret_tensor(buf158, (4, 1, 17), (17, 17, 1), 0), buf162, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf163, (4, 128, 17), (2176, 17, 1))
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(reinterpret_tensor(buf159, (4, 1, 17), (17, 17, 1), 0), buf207, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf208, (4, 128, 17), (2176, 17, 1))
        buf164 = empty_strided_cuda((4, 128, 17), (2176, 17, 1), torch.bool)
        buf165 = buf163; del buf163  # reuse
        buf209 = empty_strided_cuda((4, 128, 17), (2176, 17, 1), torch.bool)
        buf210 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_64, x_65, x_80, x_81], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_29.run(buf165, buf210, primals_61, buf164, buf209, 8704, grid=grid(8704), stream=stream0)
        del primals_61
        buf166 = empty_strided_cuda((128, 1, 1), (1, 128, 128), torch.float32)
        buf167 = reinterpret_tensor(buf166, (128, 1, 1), (1, 1, 1), 0); del buf166  # reuse
        buf168 = empty_strided_cuda((128, 32, 41), (1312, 41, 1), torch.float32)
        buf211 = empty_strided_cuda((128, 32, 41), (1312, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_17, _weight_norm_25], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_28.run(buf167, primals_63, primals_62, buf168, buf211, 128, 1312, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf165, buf168, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf169, (4, 128, 9), (1152, 9, 1))
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf210, buf211, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf212, (4, 128, 9), (1152, 9, 1))
        buf170 = empty_strided_cuda((4, 128, 9), (1152, 9, 1), torch.bool)
        buf171 = buf169; del buf169  # reuse
        buf213 = empty_strided_cuda((4, 128, 9), (1152, 9, 1), torch.bool)
        buf214 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_66, x_67, x_82, x_83], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_39.run(buf171, buf214, primals_64, buf170, buf213, 4608, grid=grid(4608), stream=stream0)
        del primals_64
        buf172 = empty_strided_cuda((256, 1, 1), (1, 256, 256), torch.float32)
        buf173 = reinterpret_tensor(buf172, (256, 1, 1), (1, 1, 1), 0); del buf172  # reuse
        buf174 = empty_strided_cuda((256, 8, 41), (328, 41, 1), torch.float32)
        buf215 = empty_strided_cuda((256, 8, 41), (328, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_18, _weight_norm_26], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_30.run(buf173, primals_66, primals_65, buf174, buf215, 256, 328, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf171, buf174, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf175, (4, 256, 5), (1280, 5, 1))
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf214, buf215, stride=(2,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf216, (4, 256, 5), (1280, 5, 1))
        buf176 = empty_strided_cuda((4, 256, 5), (1280, 5, 1), torch.bool)
        buf177 = buf175; del buf175  # reuse
        buf217 = empty_strided_cuda((4, 256, 5), (1280, 5, 1), torch.bool)
        buf218 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_68, x_69, x_84, x_85], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_40.run(buf177, buf218, primals_67, buf176, buf217, 5120, grid=grid(5120), stream=stream0)
        del primals_67
        buf178 = empty_strided_cuda((512, 1, 1), (1, 512, 512), torch.float32)
        buf179 = reinterpret_tensor(buf178, (512, 1, 1), (1, 1, 1), 0); del buf178  # reuse
        buf180 = empty_strided_cuda((512, 16, 41), (656, 41, 1), torch.float32)
        buf219 = empty_strided_cuda((512, 16, 41), (656, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_19, _weight_norm_27], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_32.run(buf179, primals_69, primals_68, buf180, buf219, 512, 656, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf177, buf180, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf181, (4, 512, 2), (1024, 2, 1))
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf218, buf219, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf220, (4, 512, 2), (1024, 2, 1))
        buf182 = empty_strided_cuda((4, 512, 2), (1024, 2, 1), torch.bool)
        buf183 = buf181; del buf181  # reuse
        buf221 = empty_strided_cuda((4, 512, 2), (1024, 2, 1), torch.bool)
        buf222 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_70, x_71, x_86, x_87], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_41.run(buf183, buf222, primals_70, buf182, buf221, 4096, grid=grid(4096), stream=stream0)
        del primals_70
        buf184 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf185 = reinterpret_tensor(buf184, (1024, 1, 1), (1, 1, 1), 0); del buf184  # reuse
        buf186 = empty_strided_cuda((1024, 32, 41), (1312, 41, 1), torch.float32)
        buf223 = empty_strided_cuda((1024, 32, 41), (1312, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_20, _weight_norm_28], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_34.run(buf185, primals_72, primals_71, buf186, buf223, 1024, 1312, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf183, buf186, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf187, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf222, buf223, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf224, (4, 1024, 1), (1024, 1, 1))
        buf188 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf189 = buf187; del buf187  # reuse
        buf225 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf226 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_72, x_73, x_88, x_89], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf189, buf226, primals_73, buf188, buf225, 4096, grid=grid(4096), stream=stream0)
        del primals_73
        buf190 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf191 = reinterpret_tensor(buf190, (1024, 1, 1), (1, 1, 1), 0); del buf190  # reuse
        buf192 = empty_strided_cuda((1024, 64, 41), (2624, 41, 1), torch.float32)
        buf227 = empty_strided_cuda((1024, 64, 41), (2624, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_21, _weight_norm_29], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_35.run(buf191, primals_75, primals_74, buf192, buf227, 1024, 2624, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf189, buf192, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf193, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf228, (4, 1024, 1), (1024, 1, 1))
        buf194 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf195 = buf193; del buf193  # reuse
        buf229 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf230 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [x_74, x_75, x_90, x_91], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf195, buf230, primals_76, buf194, buf229, 4096, grid=grid(4096), stream=stream0)
        del primals_76
        buf196 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf197 = reinterpret_tensor(buf196, (1024, 1, 1), (1, 1, 1), 0); del buf196  # reuse
        buf198 = empty_strided_cuda((1024, 1024, 5), (5120, 5, 1), torch.float32)
        buf231 = empty_strided_cuda((1024, 1024, 5), (5120, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_22, _weight_norm_30], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_36.run(buf197, primals_78, primals_77, buf198, buf231, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf195, buf198, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf199, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf230, buf231, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf232, (4, 1024, 1), (1024, 1, 1))
        buf200 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf201 = buf199; del buf199  # reuse
        buf233 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf234 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_76, x_77, x_92, x_93], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf201, buf234, primals_79, buf200, buf233, 4096, grid=grid(4096), stream=stream0)
        del primals_79
        buf202 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf203 = buf202; del buf202  # reuse
        buf204 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        buf235 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_23, _weight_norm_31], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_37.run(buf203, primals_81, primals_80, buf204, buf235, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf201, buf204, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf205, (4, 1, 1), (1, 1, 1))
        buf206 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf206, primals_82, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [x_94], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf234, buf235, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf236, (4, 1, 1), (1, 1, 1))
        buf237 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [x_94], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf237, primals_82, 4, grid=grid(4), stream=stream0)
        del primals_82
    return (reinterpret_tensor(buf46, (4, 1), (1, 1), 0), reinterpret_tensor(buf126, (4, 1), (1, 1), 0), reinterpret_tensor(buf206, (4, 1), (1, 1), 0), reinterpret_tensor(buf77, (4, 1), (1, 1), 0), reinterpret_tensor(buf157, (4, 1), (1, 1), 0), reinterpret_tensor(buf237, (4, 1), (1, 1), 0), buf5, buf11, buf17, buf23, buf29, buf35, buf41, buf46, buf85, buf91, buf97, buf103, buf109, buf115, buf121, buf126, buf165, buf171, buf177, buf183, buf189, buf195, buf201, buf206, buf50, buf54, buf58, buf62, buf66, buf70, buf74, buf77, buf130, buf134, buf138, buf142, buf146, buf150, buf154, buf157, buf210, buf214, buf218, buf222, buf226, buf230, buf234, buf237, buf47, buf51, buf55, buf59, buf63, buf67, buf71, buf75, buf127, buf131, buf135, buf139, buf143, buf147, buf151, buf155, buf207, buf211, buf215, buf219, buf223, buf227, buf231, buf235, primals_2, primals_3, primals_5, primals_7, primals_8, primals_11, primals_12, primals_15, primals_16, primals_19, primals_20, primals_23, primals_24, primals_27, primals_28, primals_31, primals_32, primals_34, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_48, primals_50, primals_51, primals_53, primals_54, primals_56, primals_57, primals_59, primals_60, primals_62, primals_63, primals_65, primals_66, primals_68, primals_69, primals_71, primals_72, primals_74, primals_75, primals_77, primals_78, primals_80, primals_81, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf20, buf22, buf23, buf25, buf26, buf28, buf29, buf31, buf32, buf34, buf35, buf37, buf38, buf40, buf41, buf43, buf44, buf47, buf49, buf50, buf51, buf53, buf54, buf55, buf57, buf58, buf59, buf61, buf62, buf63, buf65, buf66, buf67, buf69, buf70, buf71, buf73, buf74, buf75, reinterpret_tensor(buf78, (4, 1, 33), (33, 33, 1), 0), reinterpret_tensor(buf79, (4, 1, 33), (33, 33, 1), 0), buf81, buf82, buf84, buf85, buf87, buf88, buf90, buf91, buf93, buf94, buf96, buf97, buf99, buf100, buf102, buf103, buf105, buf106, buf108, buf109, buf111, buf112, buf114, buf115, buf117, buf118, buf120, buf121, buf123, buf124, buf127, buf129, buf130, buf131, buf133, buf134, buf135, buf137, buf138, buf139, buf141, buf142, buf143, buf145, buf146, buf147, buf149, buf150, buf151, buf153, buf154, buf155, reinterpret_tensor(buf158, (4, 1, 17), (17, 17, 1), 0), reinterpret_tensor(buf159, (4, 1, 17), (17, 17, 1), 0), buf161, buf162, buf164, buf165, buf167, buf168, buf170, buf171, buf173, buf174, buf176, buf177, buf179, buf180, buf182, buf183, buf185, buf186, buf188, buf189, buf191, buf192, buf194, buf195, buf197, buf198, buf200, buf201, buf203, buf204, buf207, buf209, buf210, buf211, buf213, buf214, buf215, buf217, buf218, buf219, buf221, buf222, buf223, buf225, buf226, buf227, buf229, buf230, buf231, buf233, buf234, buf235, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 1, 15), (15, 15, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((15, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 8, 41), (328, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((328, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((512, 16, 41), (656, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1024, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1024, 64, 41), (2624, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1024, 1024, 5), (5120, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((5120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, 1024, 3), (3072, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 1, 15), (15, 15, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, 8, 41), (328, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, 16, 41), (656, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1024, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1024, 64, 41), (2624, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1024, 1024, 5), (5120, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, 1024, 3), (3072, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, 1, 15), (15, 15, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, 8, 41), (328, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, 16, 41), (656, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1024, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1024, 64, 41), (2624, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1024, 1024, 5), (5120, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1, 1024, 3), (3072, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
