# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/52/c52m3ltwitjosbbgjues3qvbdnms5qk36qvawn62xdyyo3vy3pv2.py
# Topologically Sorted Source Nodes: [norm, denom, normalized_weight], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div]
# Source node to ATen node mapping:
#   denom => clamp_min
#   norm => pow_1, pow_2, sum_1
#   normalized_weight => div
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_15, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-12), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_15, %clamp_min), kwargs = {})
triton_red_fused_clamp_min_div_linalg_vector_norm_0 = async_compile.triton('triton_red_fused_clamp_min_div_linalg_vector_norm_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clamp_min_div_linalg_vector_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_clamp_min_div_linalg_vector_norm_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp6 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp7 = 1e-12
        tmp8 = triton_helpers.maximum(tmp5, tmp7)
        tmp9 = tmp6 / tmp8
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/rg/crgvjpcauu6m5egeo4tv4s5nywh4764w2naywoum7rpi6buwdj5s.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%unsqueeze, [1, 4], [1, 2], [0, 2]), kwargs = {})
triton_poi_fused_avg_pool2d_1 = async_compile.triton('triton_poi_fused_avg_pool2d_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/6x/c6xsfzhifyy2clsqarh7icn24usyzf3gknopcjlky3dryg3o53mk.py
# Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm => pow_10, pow_9, sum_5
# Graph fragment:
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_63, 2), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [1, 2, 3], True), kwargs = {})
#   %pow_10 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_5, 0.5), kwargs = {})
triton_poi_fused__weight_norm_interface_2 = async_compile.triton('triton_poi_fused__weight_norm_interface_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 5*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 5*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 5*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4 + 5*x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dm/cdm3uokotllxm55fezs3k32wjpumyrigr5h7cqokdkqr2cavgtbj.py
# Topologically Sorted Source Nodes: [_weight_norm_1], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_1 => div_5, mul_30, pow_11, pow_12, sum_6
# Graph fragment:
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_66, 2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1, 2, 3], True), kwargs = {})
#   %pow_12 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_6, 0.5), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_65, %pow_12), kwargs = {})
#   %mul_30 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_66, %div_5), kwargs = {})
triton_per_fused__weight_norm_interface_3 = async_compile.triton('triton_per_fused__weight_norm_interface_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 160
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 160*x0), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 160*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qn/cqni5dd4qqzeyp34xohumduesrqlvt7ry7bjxrr4xy72m5urstek.py
# Topologically Sorted Source Nodes: [_weight_norm_2], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_2 => div_6, mul_32, pow_13, pow_14, sum_7
# Graph fragment:
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_69, 2), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_13, [1, 2, 3], True), kwargs = {})
#   %pow_14 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_7, 0.5), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_68, %pow_14), kwargs = {})
#   %mul_32 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_69, %div_6), kwargs = {})
triton_per_fused__weight_norm_interface_4 = async_compile.triton('triton_per_fused__weight_norm_interface_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 640
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 640*x0), rmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 640*x0), tmp9, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/zn/cznirevjginvk7f756mlarwq4jr6timzzfnbxas26n5n4txafj5o.py
# Topologically Sorted Source Nodes: [_weight_norm_3], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_3 => div_7, mul_34, pow_15, pow_16, sum_8
# Graph fragment:
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_72, 2), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_15, [1, 2, 3], True), kwargs = {})
#   %pow_16 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_8, 0.5), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_71, %pow_16), kwargs = {})
#   %mul_34 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_72, %div_7), kwargs = {})
triton_red_fused__weight_norm_interface_5 = async_compile.triton('triton_red_fused__weight_norm_interface_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_5(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2560
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
        tmp0 = tl.load(in_ptr0 + (r1 + 2560*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp6 = tl.load(in_ptr0 + (r1 + 2560*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 / tmp5
        tmp9 = tmp6 * tmp8
        tl.store(out_ptr0 + (r1 + 2560*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ml/cmlxmjcrenx52u4p4eugla6z2t6hnz2dl7pk4esue5xcuhcyxzlx.py
# Topologically Sorted Source Nodes: [_weight_norm_4, norm_4, denom_4, normalized_weight_4], Original ATen: [aten._weight_norm_interface, aten.linalg_vector_norm, aten.clamp_min, aten.div]
# Source node to ATen node mapping:
#   _weight_norm_4 => div_8, mul_36, pow_17, pow_18, sum_9
#   denom_4 => clamp_min_4
#   norm_4 => pow_19, pow_20, sum_10
#   normalized_weight_4 => div_9
# Graph fragment:
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_75, 2), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_17, [1, 2, 3], True), kwargs = {})
#   %pow_18 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_9, 0.5), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_74, %pow_18), kwargs = {})
#   %mul_36 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_75, %div_8), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_36, 2.0), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_19, [1, 2, 3], True), kwargs = {})
#   %pow_20 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_10, 0.5), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_20, 1e-12), kwargs = {})
#   %div_9 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_36, %clamp_min_4), kwargs = {})
triton_red_fused__weight_norm_interface_clamp_min_div_linalg_vector_norm_6 = async_compile.triton('triton_red_fused__weight_norm_interface_clamp_min_div_linalg_vector_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_clamp_min_div_linalg_vector_norm_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_clamp_min_div_linalg_vector_norm_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2048
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
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp6 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp8 / tmp5
        tmp10 = tmp6 * tmp9
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp10, rmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = libdevice.sqrt(tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp15, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp16 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = 1e-12
        tmp18 = triton_helpers.maximum(tmp15, tmp17)
        tmp19 = tmp16 / tmp18
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp19, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/45/c45mzpzm2qtsrsj3rhp3iljlmtdgdpv24m7xf6pjttxy6gzgfh52.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_5 => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=3] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_3, [None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad1d_7 = async_compile.triton('triton_poi_fused_reflection_pad1d_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = xindex // 66
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u3/cu3av3q4ry4cs36s4kpkjy6zkdhlf2qiuqxfc2iiodm254nnjepp.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_7 => _unsafe_index_1
# Graph fragment:
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_3, [None, None, %sub_3]), kwargs = {})
triton_poi_fused_reflection_pad1d_8 = async_compile.triton('triton_poi_fused_reflection_pad1d_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 260
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 65)
    x1 = xindex // 65
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5i/c5imujhjmg5u4nq5k4w6iusei545oukzhny4cy65ij4zdrmcai42.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_9 => _unsafe_index_2
# Graph fragment:
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_3, [None, None, %sub_5]), kwargs = {})
triton_poi_fused_reflection_pad1d_9 = async_compile.triton('triton_poi_fused_reflection_pad1d_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 70)
    x1 = xindex // 70
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e3/ce3ajnhqjfs4fj5agb62jzz5xy4fdwkvwj3xrzwyoro3bmb34lmg.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => gt, mul, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul), kwargs = {})
triton_poi_fused_convolution_leaky_relu_10 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_10(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/qp/cqpxmwtk65ryhgxbdflyql4t3m3cw6ukwyj4f6awssti5abplx7y.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_1 => avg_pool2d_1
# Graph fragment:
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%unsqueeze_1, [1, 4], [1, 2], [0, 2]), kwargs = {})
triton_poi_fused_avg_pool2d_11 = async_compile.triton('triton_poi_fused_avg_pool2d_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/sf/csfo7vtufrypjllveh7fefq3sqwitdukplht6jiamuuirm5rqtiw.py
# Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm => div_4, mul_28
# Graph fragment:
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_62, %pow_10), kwargs = {})
#   %mul_28 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_63, %div_4), kwargs = {})
triton_poi_fused__weight_norm_interface_12 = async_compile.triton('triton_poi_fused__weight_norm_interface_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 5
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ho/choc6c3ge3w5xxupywz6p6bpupfil6sykdgtyiizajcyy6bw3mwz.py
# Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_14 => convolution_7
#   input_15 => gt_6, mul_7, where_6
# Graph fragment:
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%squeeze, %primals_17, %primals_18, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt_6 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_7, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_7, 0.1), kwargs = {})
#   %where_6 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %convolution_7, %mul_7), kwargs = {})
triton_poi_fused_convolution_leaky_relu_13 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_13(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 33) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fr/cfrsayzsv75ycylik4mvz3besslufnuvvcye4ysnn22iifwte32h.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_2 => avg_pool2d_2
# Graph fragment:
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%unsqueeze_2, [1, 4], [1, 2], [0, 2]), kwargs = {})
triton_poi_fused_avg_pool2d_14 = async_compile.triton('triton_poi_fused_avg_pool2d_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 9)
    x1 = xindex // 9
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tmp1 & tmp3
    tmp5 = (-2) + 2*x0
    tmp6 = tmp5 >= tmp0
    tmp7 = tl.full([1], 17, tl.int64)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tmp4 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-2) + 2*x0 + 17*x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = (-1) + 2*x0
    tmp13 = tmp12 >= tmp0
    tmp14 = tmp12 < tmp7
    tmp15 = tmp13 & tmp14
    tmp16 = tmp4 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-1) + 2*x0 + 17*x1), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 2*x0
    tmp20 = tmp19 >= tmp0
    tmp21 = tmp19 < tmp7
    tmp22 = tmp20 & tmp21
    tmp23 = tmp4 & tmp22
    tmp24 = tl.load(in_ptr0 + (2*x0 + 17*x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 1 + 2*x0
    tmp27 = tmp26 >= tmp0
    tmp28 = tmp26 < tmp7
    tmp29 = tmp27 & tmp28
    tmp30 = tmp4 & tmp29
    tmp31 = tl.load(in_ptr0 + (1 + 2*x0 + 17*x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = 2 + ((-2)*x0) + ((19) * ((19) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (19)))
    tmp34 = tmp32 / tmp33
    tl.store(out_ptr0 + (x2), tmp34, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zk/czkv76jsfbglzwhi6cp5ui5brwlsf2jgcnyydzwvk44wfennzb7n.py
# Topologically Sorted Source Nodes: [input_62, input_63], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_62 => convolution_33
#   input_63 => gt_28, mul_39, where_28
# Graph fragment:
#   %convolution_33 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_11, %mul_38, %primals_80, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_28 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_33, 0), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_33, 0.1), kwargs = {})
#   %where_28 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_28, %convolution_33, %mul_39), kwargs = {})
triton_poi_fused_convolution_leaky_relu_15 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_15(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 24) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n4/cn4k3ifl76t7epg7gx2mxu6nqorsgpl2xignz6jpis5qjvapg6x4.py
# Topologically Sorted Source Nodes: [input_71, input_72], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_71 => convolution_38
#   input_72 => gt_32, mul_49, where_32
# Graph fragment:
#   %convolution_38 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_14, %mul_48, %primals_96, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_32 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_38, 0), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_38, 0.1), kwargs = {})
#   %where_32 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_32, %convolution_38, %mul_49), kwargs = {})
triton_poi_fused_convolution_leaky_relu_16 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_16(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 25) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j4/cj4w7x4uavdeszve22pramaccffyw6rt64be7rm3tfk6tu3eczrp.py
# Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_80 => convolution_43
#   input_81 => gt_36, mul_59, where_36
# Graph fragment:
#   %convolution_43 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_17, %mul_58, %primals_112, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_36 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_43, 0), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_43, 0.1), kwargs = {})
#   %where_36 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_36, %convolution_43, %mul_59), kwargs = {})
triton_poi_fused_convolution_leaky_relu_17 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_17(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 28) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ym/cym4l3wp4twdxdxpuvoxsqjmxclw6ohw7tdg7oftggvymryxfn7v.py
# Topologically Sorted Source Nodes: [input_89, input_90], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_89 => convolution_48
#   input_90 => gt_40, mul_69, where_40
# Graph fragment:
#   %convolution_48 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_20, %mul_68, %primals_128, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_40 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_48, 0), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_48, 0.1), kwargs = {})
#   %where_40 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_40, %convolution_48, %mul_69), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_18(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 22) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vw/cvwips6b75tbvg7afllwrsgjetkiuigq3wl2n6eh5wrqon7iy2vg.py
# Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_3 => convolution_1
#   input_4 => gt_1, mul_1, where_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where, %primals_4, %primals_5, [4], [20], [1], False, [0], 4), kwargs = {})
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.1), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_1, %mul_1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_19 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_19(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/du/cdusqwgn7wya6ovqhichv4zr7w2jpkp3qy5e2yqrhtzr77wxqm32.py
# Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_27 => convolution_14
#   input_28 => gt_12, mul_14, where_12
# Graph fragment:
#   %convolution_14 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%squeeze_1, %primals_32, %primals_33, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt_12 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_14, 0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_14, 0.1), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %convolution_14, %mul_14), kwargs = {})
triton_poi_fused_convolution_leaky_relu_20 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_20(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 17) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ym/cymvxtxgsasxsftorpmsulhh5yp2hvkij3nuh7zeaaoqc72tujzt.py
# Topologically Sorted Source Nodes: [input_64, input_65], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_64 => convolution_34
#   input_65 => gt_29, mul_41, where_29
# Graph fragment:
#   %convolution_34 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_28, %mul_40, %primals_83, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_29 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_34, 0), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_34, 0.1), kwargs = {})
#   %where_29 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_29, %convolution_34, %mul_41), kwargs = {})
triton_poi_fused_convolution_leaky_relu_21 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_21(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 9) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kj/ckjd3vqkjatrnlol3ke76oidis5zckuicpyvffny4ircypi6vv4c.py
# Topologically Sorted Source Nodes: [input_73, input_74], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_73 => convolution_39
#   input_74 => gt_33, mul_51, where_33
# Graph fragment:
#   %convolution_39 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_32, %mul_50, %primals_99, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_33 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_39, 0), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_39, 0.1), kwargs = {})
#   %where_33 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_33, %convolution_39, %mul_51), kwargs = {})
triton_poi_fused_convolution_leaky_relu_22 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_22(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 10) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qv/cqvankzx47sdmj5tiypmonpetrej4tni442to4d7fprb4soslo3b.py
# Topologically Sorted Source Nodes: [input_82, input_83], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_82 => convolution_44
#   input_83 => gt_37, mul_61, where_37
# Graph fragment:
#   %convolution_44 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_36, %mul_60, %primals_115, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_37 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_44, 0), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_44, 0.1), kwargs = {})
#   %where_37 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_37, %convolution_44, %mul_61), kwargs = {})
triton_poi_fused_convolution_leaky_relu_23 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_23(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 14) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z5/cz5a6jdkzlh7ngz2yr5xxnshhd7opfkntaqyli553btwhdrvqxvs.py
# Topologically Sorted Source Nodes: [input_91, input_92], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_91 => convolution_49
#   input_92 => gt_41, mul_71, where_41
# Graph fragment:
#   %convolution_49 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_40, %mul_70, %primals_131, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_41 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_49, 0), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_49, 0.1), kwargs = {})
#   %where_41 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_41, %convolution_49, %mul_71), kwargs = {})
triton_poi_fused_convolution_leaky_relu_24 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_24(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 11) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x5/cx5nmdc23sctlohdbiaiv27pqr6fz4nslps3yfhg77nnlzb4yate.py
# Topologically Sorted Source Nodes: [input_55, input_56], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_55 => convolution_29
#   input_56 => gt_25, mul_31, where_25
# Graph fragment:
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_24, %mul_30, %primals_67, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_25 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_29, 0), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_29, 0.1), kwargs = {})
#   %where_25 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_25, %convolution_29, %mul_31), kwargs = {})
triton_poi_fused_convolution_leaky_relu_25 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_25(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 8) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/sp/cspyv4cez3qid3sxhozihtnuuhbrpf7bsxrunixosecqbzscxjum.py
# Topologically Sorted Source Nodes: [input_66, input_67], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_66 => convolution_35
#   input_67 => gt_30, mul_43, where_30
# Graph fragment:
#   %convolution_35 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_29, %mul_42, %primals_86, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_30 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_35, 0), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_35, 0.1), kwargs = {})
#   %where_30 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_30, %convolution_35, %mul_43), kwargs = {})
triton_poi_fused_convolution_leaky_relu_26 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_26(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ek/cekrcs7mf2z7uuojag3hr37yxzjsevvxh5dgurdwzzgula5uya55.py
# Topologically Sorted Source Nodes: [input_75, input_76], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_75 => convolution_40
#   input_76 => gt_34, mul_53, where_34
# Graph fragment:
#   %convolution_40 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_33, %mul_52, %primals_102, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_34 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_40, 0), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_40, 0.1), kwargs = {})
#   %where_34 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_34, %convolution_40, %mul_53), kwargs = {})
triton_poi_fused_convolution_leaky_relu_27 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_27(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 5) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zx/czxnaslrvpdyxn2mw2vg2wbd7yceeq6ap33ni6fxrghttikzkwez.py
# Topologically Sorted Source Nodes: [input_84, input_85], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_84 => convolution_45
#   input_85 => gt_38, mul_63, where_38
# Graph fragment:
#   %convolution_45 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_37, %mul_62, %primals_118, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_38 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_45, 0), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_45, 0.1), kwargs = {})
#   %where_38 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_38, %convolution_45, %mul_63), kwargs = {})
triton_poi_fused_convolution_leaky_relu_28 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_28(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 7) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uw/cuw5ig4tz5xay4iig5i26v25l2qwreb6zoh2oufvio4smjpu7ugi.py
# Topologically Sorted Source Nodes: [input_93, input_94], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_93 => convolution_50
#   input_94 => gt_42, mul_73, where_42
# Graph fragment:
#   %convolution_50 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_41, %mul_72, %primals_134, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_42 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_50, 0), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_50, 0.1), kwargs = {})
#   %where_42 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_42, %convolution_50, %mul_73), kwargs = {})
triton_poi_fused_convolution_leaky_relu_29 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_29(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 11) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p7/cp7e3drmakw4qsoxkewzz7nvgaykwmlnjrri54pmw7b45mi4r7sh.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_5 => convolution_2
#   input_6 => gt_2, mul_2, where_2
# Graph fragment:
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_1, %primals_6, %primals_7, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_2 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, 0.1), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convolution_2, %mul_2), kwargs = {})
triton_poi_fused_convolution_leaky_relu_30 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_30(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/ha/chatfscqifhkow2rn4oamits2g4mzymxaywdb2pnirvwpofbgwg6.py
# Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_29 => convolution_15
#   input_30 => gt_13, mul_15, where_13
# Graph fragment:
#   %convolution_15 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_12, %primals_34, %primals_35, [4], [20], [1], False, [0], 4), kwargs = {})
#   %gt_13 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_15, 0), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, 0.1), kwargs = {})
#   %where_13 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_13, %convolution_15, %mul_15), kwargs = {})
triton_poi_fused_convolution_leaky_relu_31 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_31(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 5) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ka/ckat6p42od6w7iy6atrpxg6jypaoucwe574vytxhkjv3jinopks6.py
# Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_57 => convolution_30
#   input_58 => gt_26, mul_33, where_26
# Graph fragment:
#   %convolution_30 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_25, %mul_32, %primals_70, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_26 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_30, 0), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_30, 0.1), kwargs = {})
#   %where_26 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_26, %convolution_30, %mul_33), kwargs = {})
triton_poi_fused_convolution_leaky_relu_32 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_32(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/gn/cgnlrranfiueynbvrxt4hfsx7t7cpz3ninmcohhymambpu4nonnb.py
# Topologically Sorted Source Nodes: [input_68, input_69, input_70], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_68 => convolution_36
#   input_69 => gt_31, mul_45, where_31
#   input_70 => add_5
# Graph fragment:
#   %convolution_36 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_30, %mul_44, %primals_89, [1, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_31 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_36, 0), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_36, 0.1), kwargs = {})
#   %where_31 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_31, %convolution_36, %mul_45), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_31, %view_12), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_33 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_33(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 3) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp7 + tmp8
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/tg/ctg7h4ebaumuhevuxq6nqncn43cfbnlmwfktgp4sl25ifb5cqky3.py
# Topologically Sorted Source Nodes: [input_77, input_78, input_79], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_77 => convolution_41
#   input_78 => gt_35, mul_55, where_35
#   input_79 => add_6
# Graph fragment:
#   %convolution_41 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_34, %mul_54, %primals_105, [1, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_35 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_41, 0), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_41, 0.1), kwargs = {})
#   %where_35 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_35, %convolution_41, %mul_55), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_35, %view_15), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_34 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_34(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 5) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp7 + tmp8
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/tc/ctcz65yjvgjzava7p6e7cyz3hpipjet43bzbpwmectb7hcbeg2t5.py
# Topologically Sorted Source Nodes: [input_86, input_87, input_88], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_86 => convolution_46
#   input_87 => gt_39, mul_65, where_39
#   input_88 => add_7
# Graph fragment:
#   %convolution_46 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_38, %mul_64, %primals_121, [1, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_39 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_46, 0), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_46, 0.1), kwargs = {})
#   %where_39 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_39, %convolution_46, %mul_65), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_39, %view_18), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_35 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_35(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 7) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp7 + tmp8
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/dw/cdwfejjgj5rojelvmptgfxuider2uzgvq4wax6gke4iqiosdvlrs.py
# Topologically Sorted Source Nodes: [input_95, input_96, input_97], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_95 => convolution_51
#   input_96 => gt_43, mul_75, where_43
#   input_97 => add_8
# Graph fragment:
#   %convolution_51 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_42, %mul_74, %primals_137, [1, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_43 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_51, 0), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_51, 0.1), kwargs = {})
#   %where_43 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_43, %convolution_51, %mul_75), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_43, %view_21), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_36 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_36(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 11) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp7 + tmp8
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/mq/cmqipiwglkjccqlflk62kqu3tdbyn5h4yhzsurepyzfaehj3fhdc.py
# Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_18 => convolution_9
#   input_19 => gt_8, mul_9, where_8
# Graph fragment:
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_7, %primals_21, %primals_22, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_8 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_9, 0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_9, 0.1), kwargs = {})
#   %where_8 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %convolution_9, %mul_9), kwargs = {})
triton_poi_fused_convolution_leaky_relu_37 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_37(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j6/cj6macukl73fpt6zywl4nnfp2ti3bpfav2tedqs4nyoqctedevoh.py
# Topologically Sorted Source Nodes: [input_42, input_43], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_42 => convolution_22
#   input_43 => gt_19, mul_22, where_19
# Graph fragment:
#   %convolution_22 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_18, %primals_49, %primals_50, [4], [20], [1], False, [0], 4), kwargs = {})
#   %gt_19 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_22, 0), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_22, 0.1), kwargs = {})
#   %where_19 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_19, %convolution_22, %mul_22), kwargs = {})
triton_poi_fused_convolution_leaky_relu_38 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_38(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/no/cnoxgmgpn5dcpk452p3xvvdrg47kg52qrngn76ftwdshxb276hd5.py
# Topologically Sorted Source Nodes: [input_59, input_60, input_61], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_59 => convolution_31
#   input_60 => gt_27, mul_35, where_27
#   input_61 => add_4
# Graph fragment:
#   %convolution_31 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_26, %mul_34, %primals_73, [1, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_27 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_31, 0), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_31, 0.1), kwargs = {})
#   %where_27 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_27, %convolution_31, %mul_35), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_27, %view_9), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_39 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_39(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp7 + tmp8
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/63/c63omnc5kyreq5llxhjg634intnnyef3wfi5el6iarugmdfwxnbh.py
# Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   out_11 => mul_47
# Graph fragment:
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_37, %view_13), kwargs = {})
triton_poi_fused_mul_40 = async_compile.triton('triton_poi_fused_mul_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_40(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/q7/cq7f6vugmqj5u3npvexjhiby6i7tolpi2ophyckw3bh4s3xwtz7b.py
# Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   out_13 => mul_57
# Graph fragment:
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_42, %view_16), kwargs = {})
triton_poi_fused_mul_41 = async_compile.triton('triton_poi_fused_mul_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_41(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tu/ctumwugtyqn6oaqoi3mhqhgad47cmlrohkenx6q6miftzgpn4sdy.py
# Topologically Sorted Source Nodes: [out_15], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   out_15 => mul_67
# Graph fragment:
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_47, %view_19), kwargs = {})
triton_poi_fused_mul_42 = async_compile.triton('triton_poi_fused_mul_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_42(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rf/crfhr5gd6e3dtb4h2daq54dov7dia5agk2dtuq5bebed3zjc72v7.py
# Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   out_17 => mul_77
# Graph fragment:
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_52, %view_22), kwargs = {})
triton_poi_fused_mul_43 = async_compile.triton('triton_poi_fused_mul_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_43(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 88
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ch/cchrdgmyqwefldx5vyn63jojsyqlhxqvvrcrt26pk6iakfjqtg7f.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_7 => convolution_3
#   input_8 => gt_3, mul_3, where_3
# Graph fragment:
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_2, %primals_8, %primals_9, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_3 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_3, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_3, 0.1), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %convolution_3, %mul_3), kwargs = {})
triton_poi_fused_convolution_leaky_relu_44 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_44(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tx/ctxjji5wztccg7ckf4e24r4zvg7stoiu2zkj4yytp6iaxzz76quy.py
# Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_31 => convolution_16
#   input_32 => gt_14, mul_16, where_14
# Graph fragment:
#   %convolution_16 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_13, %primals_36, %primals_37, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_14 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_16, 0), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_16, 0.1), kwargs = {})
#   %where_14 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %convolution_16, %mul_16), kwargs = {})
triton_poi_fused_convolution_leaky_relu_45 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_45', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_45(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 2) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wr/cwrhw6gobqb2qiwdrx3elschclbusfjlf4zv4sgokjnc4nsihcqx.py
# Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_44 => convolution_23
#   input_45 => gt_20, mul_23, where_20
# Graph fragment:
#   %convolution_23 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_19, %primals_51, %primals_52, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_20 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_23, 0), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_23, 0.1), kwargs = {})
#   %where_20 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_20, %convolution_23, %mul_23), kwargs = {})
triton_poi_fused_convolution_leaky_relu_46 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_46(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sq/csqaoesnn6xsjuvujg3edfopmaw3iu4ir3rtellfblptx6w3mfz2.py
# Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_10 => gt_4, mul_4, where_4
#   input_9 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_3, %primals_10, %primals_11, [1], [20], [1], False, [0], 16), kwargs = {})
#   %gt_4 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_4, 0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_4, 0.1), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %convolution_4, %mul_4), kwargs = {})
triton_poi_fused_convolution_leaky_relu_47 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_47', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_47(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x2), tmp4, None)
    tl.store(in_out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/2c/c2cijkcgk6ukhgaj6qjo2kcwchwggexil7x4lt4jqql7j3icjcgz.py
# Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   input_11 => convolution_5
#   input_12 => gt_5, mul_5, where_5
#   input_13 => add
# Graph fragment:
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_4, %primals_12, %primals_13, [1], [2], [1], False, [0], 1), kwargs = {})
#   %gt_5 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_5, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_5, 0.1), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %convolution_5, %mul_5), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_5, %view), kwargs = {})
triton_poi_fused_add_convolution_leaky_relu_48 = async_compile.triton('triton_poi_fused_add_convolution_leaky_relu_48', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_leaky_relu_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_leaky_relu_48(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp7 + tmp8
    tl.store(out_ptr0 + (x2), tmp4, None)
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr1 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/xw/cxwi35y73drczhd54h6enzeki6oitt5kcwgakibi3tkg24xs5avg.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   out_1 => mul_6
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_6, %view_1), kwargs = {})
triton_poi_fused_mul_49 = async_compile.triton('triton_poi_fused_mul_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_49(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141 = args
    args.clear()
    assert_size_stride(primals_1, (128, 1, 15), (15, 15, 1))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (4, 1, 64), (64, 64, 1))
    assert_size_stride(primals_4, (128, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (256, 8, 41), (328, 41, 1))
    assert_size_stride(primals_7, (256, ), (1, ))
    assert_size_stride(primals_8, (512, 16, 41), (656, 41, 1))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (1024, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, 1024, 5), (5120, 5, 1))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_15, (1, 1024, 3), (3072, 3, 1))
    assert_size_stride(primals_16, (1, ), (1, ))
    assert_size_stride(primals_17, (128, 1, 15), (15, 15, 1))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (256, 8, 41), (328, 41, 1))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (512, 16, 41), (656, 41, 1))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (1024, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_26, (1024, ), (1, ))
    assert_size_stride(primals_27, (1024, 1024, 5), (5120, 5, 1))
    assert_size_stride(primals_28, (1024, ), (1, ))
    assert_size_stride(primals_29, (1024, ), (1, ))
    assert_size_stride(primals_30, (1, 1024, 3), (3072, 3, 1))
    assert_size_stride(primals_31, (1, ), (1, ))
    assert_size_stride(primals_32, (128, 1, 15), (15, 15, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (256, 8, 41), (328, 41, 1))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_38, (512, 16, 41), (656, 41, 1))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (1024, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_42, (1024, 1024, 5), (5120, 5, 1))
    assert_size_stride(primals_43, (1024, ), (1, ))
    assert_size_stride(primals_44, (1024, ), (1, ))
    assert_size_stride(primals_45, (1, 1024, 3), (3072, 3, 1))
    assert_size_stride(primals_46, (1, ), (1, ))
    assert_size_stride(primals_47, (128, 1, 15), (15, 15, 1))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (256, 8, 41), (328, 41, 1))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_53, (512, 16, 41), (656, 41, 1))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (1024, 32, 41), (1312, 41, 1))
    assert_size_stride(primals_56, (1024, ), (1, ))
    assert_size_stride(primals_57, (1024, 1024, 5), (5120, 5, 1))
    assert_size_stride(primals_58, (1024, ), (1, ))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_60, (1, 1024, 3), (3072, 3, 1))
    assert_size_stride(primals_61, (1, ), (1, ))
    assert_size_stride(primals_62, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_63, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_64, (32, ), (1, ))
    assert_size_stride(primals_65, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_66, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_68, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_69, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_72, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_73, (1024, ), (1, ))
    assert_size_stride(primals_74, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_75, (1, 1024, 2, 1), (2048, 2, 1, 1))
    assert_size_stride(primals_76, (1024, ), (1, ))
    assert_size_stride(primals_77, (1, ), (1, ))
    assert_size_stride(primals_78, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_79, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_80, (32, ), (1, ))
    assert_size_stride(primals_81, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_82, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_85, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_88, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_89, (1024, ), (1, ))
    assert_size_stride(primals_90, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_91, (1, 1024, 2, 1), (2048, 2, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1, ), (1, ))
    assert_size_stride(primals_94, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_95, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_96, (32, ), (1, ))
    assert_size_stride(primals_97, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_98, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_101, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_104, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_105, (1024, ), (1, ))
    assert_size_stride(primals_106, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_107, (1, 1024, 2, 1), (2048, 2, 1, 1))
    assert_size_stride(primals_108, (1024, ), (1, ))
    assert_size_stride(primals_109, (1, ), (1, ))
    assert_size_stride(primals_110, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_111, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_112, (32, ), (1, ))
    assert_size_stride(primals_113, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_114, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_117, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_119, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_120, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_122, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_123, (1, 1024, 2, 1), (2048, 2, 1, 1))
    assert_size_stride(primals_124, (1024, ), (1, ))
    assert_size_stride(primals_125, (1, ), (1, ))
    assert_size_stride(primals_126, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_127, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_128, (32, ), (1, ))
    assert_size_stride(primals_129, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_130, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_133, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_136, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_137, (1024, ), (1, ))
    assert_size_stride(primals_138, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_139, (1, 1024, 2, 1), (2048, 2, 1, 1))
    assert_size_stride(primals_140, (1024, ), (1, ))
    assert_size_stride(primals_141, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf0, (4, 128, 64), (8192, 64, 1))
        buf19 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf20 = buf19; del buf19  # reuse
        buf21 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [norm, denom, normalized_weight], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_clamp_min_div_linalg_vector_norm_0.run(buf20, primals_15, buf21, 1, 3072, grid=grid(1), stream=stream0)
        buf24 = empty_strided_cuda((4, 1, 1, 33), (33, 33, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_1.run(primals_3, buf24, 132, grid=grid(132), stream=stream0)
        buf44 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf45 = buf44; del buf44  # reuse
        buf46 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [norm_1, denom_1, normalized_weight_1], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_clamp_min_div_linalg_vector_norm_0.run(buf45, primals_30, buf46, 1, 3072, grid=grid(1), stream=stream0)
        buf69 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf70 = buf69; del buf69  # reuse
        buf71 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [norm_2, denom_2, normalized_weight_2], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_clamp_min_div_linalg_vector_norm_0.run(buf70, primals_45, buf71, 1, 3072, grid=grid(1), stream=stream0)
        buf94 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf95 = buf94; del buf94  # reuse
        buf96 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [norm_3, denom_3, normalized_weight_3], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_clamp_min_div_linalg_vector_norm_0.run(buf95, primals_60, buf96, 1, 3072, grid=grid(1), stream=stream0)
        buf99 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_2.run(primals_63, buf99, 32, grid=grid(32), stream=stream0)
        buf104 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf105 = reinterpret_tensor(buf104, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf104  # reuse
        buf106 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_1], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf105, primals_66, primals_65, buf106, 128, 160, grid=grid(128), stream=stream0)
        buf110 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf111 = reinterpret_tensor(buf110, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf110  # reuse
        buf112 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_2], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf111, primals_69, primals_68, buf112, 512, 640, grid=grid(512), stream=stream0)
        buf116 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf117 = reinterpret_tensor(buf116, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf116  # reuse
        buf118 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_3], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf117, primals_72, primals_71, buf118, 1024, 2560, grid=grid(1024), stream=stream0)
        buf122 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf123 = buf122; del buf122  # reuse
        buf124 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        buf126 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf127 = buf126; del buf126  # reuse
        buf128 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_4, norm_4, denom_4, normalized_weight_4], Original ATen: [aten._weight_norm_interface, aten.linalg_vector_norm, aten.clamp_min, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_clamp_min_div_linalg_vector_norm_6.run(buf123, buf127, primals_75, primals_74, buf124, buf128, 1, 2048, grid=grid(1), stream=stream0)
        buf131 = empty_strided_cuda((4, 1, 66), (66, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_7.run(primals_3, buf131, 264, grid=grid(264), stream=stream0)
        buf132 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_5], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_2.run(primals_79, buf132, 32, grid=grid(32), stream=stream0)
        buf137 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf138 = reinterpret_tensor(buf137, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf137  # reuse
        buf139 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_6], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf138, primals_82, primals_81, buf139, 128, 160, grid=grid(128), stream=stream0)
        buf143 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf144 = reinterpret_tensor(buf143, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf143  # reuse
        buf145 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_7], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf144, primals_85, primals_84, buf145, 512, 640, grid=grid(512), stream=stream0)
        buf149 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf150 = reinterpret_tensor(buf149, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf149  # reuse
        buf151 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_8], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf150, primals_88, primals_87, buf151, 1024, 2560, grid=grid(1024), stream=stream0)
        buf155 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf156 = buf155; del buf155  # reuse
        buf157 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        buf159 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf160 = buf159; del buf159  # reuse
        buf161 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_9, norm_5, denom_5, normalized_weight_5], Original ATen: [aten._weight_norm_interface, aten.linalg_vector_norm, aten.clamp_min, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_clamp_min_div_linalg_vector_norm_6.run(buf156, buf160, primals_91, primals_90, buf157, buf161, 1, 2048, grid=grid(1), stream=stream0)
        buf164 = empty_strided_cuda((4, 1, 65), (65, 65, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_8.run(primals_3, buf164, 260, grid=grid(260), stream=stream0)
        buf165 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_10], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_2.run(primals_95, buf165, 32, grid=grid(32), stream=stream0)
        buf170 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf171 = reinterpret_tensor(buf170, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf170  # reuse
        buf172 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_11], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf171, primals_98, primals_97, buf172, 128, 160, grid=grid(128), stream=stream0)
        buf176 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf177 = reinterpret_tensor(buf176, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf176  # reuse
        buf178 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_12], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf177, primals_101, primals_100, buf178, 512, 640, grid=grid(512), stream=stream0)
        buf182 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf183 = reinterpret_tensor(buf182, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf182  # reuse
        buf184 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_13], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf183, primals_104, primals_103, buf184, 1024, 2560, grid=grid(1024), stream=stream0)
        buf188 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf189 = buf188; del buf188  # reuse
        buf190 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        buf192 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf193 = buf192; del buf192  # reuse
        buf194 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_14, norm_6, denom_6, normalized_weight_6], Original ATen: [aten._weight_norm_interface, aten.linalg_vector_norm, aten.clamp_min, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_clamp_min_div_linalg_vector_norm_6.run(buf189, buf193, primals_107, primals_106, buf190, buf194, 1, 2048, grid=grid(1), stream=stream0)
        buf197 = empty_strided_cuda((4, 1, 70), (70, 70, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_9.run(primals_3, buf197, 280, grid=grid(280), stream=stream0)
        buf198 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_15], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_2.run(primals_111, buf198, 32, grid=grid(32), stream=stream0)
        buf203 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf204 = reinterpret_tensor(buf203, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf203  # reuse
        buf205 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_16], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf204, primals_114, primals_113, buf205, 128, 160, grid=grid(128), stream=stream0)
        buf209 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf210 = reinterpret_tensor(buf209, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf209  # reuse
        buf211 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_17], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf210, primals_117, primals_116, buf211, 512, 640, grid=grid(512), stream=stream0)
        buf215 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf216 = reinterpret_tensor(buf215, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf215  # reuse
        buf217 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_18], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf216, primals_120, primals_119, buf217, 1024, 2560, grid=grid(1024), stream=stream0)
        buf221 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf222 = buf221; del buf221  # reuse
        buf223 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        buf225 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf226 = buf225; del buf225  # reuse
        buf227 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_19, norm_7, denom_7, normalized_weight_7], Original ATen: [aten._weight_norm_interface, aten.linalg_vector_norm, aten.clamp_min, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_clamp_min_div_linalg_vector_norm_6.run(buf222, buf226, primals_123, primals_122, buf223, buf227, 1, 2048, grid=grid(1), stream=stream0)
        buf230 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_20], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_2.run(primals_127, buf230, 32, grid=grid(32), stream=stream0)
        buf235 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf236 = reinterpret_tensor(buf235, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf235  # reuse
        buf237 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_21], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf236, primals_130, primals_129, buf237, 128, 160, grid=grid(128), stream=stream0)
        buf241 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf242 = reinterpret_tensor(buf241, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf241  # reuse
        buf243 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_22], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf242, primals_133, primals_132, buf243, 512, 640, grid=grid(512), stream=stream0)
        buf247 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf248 = reinterpret_tensor(buf247, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf247  # reuse
        buf249 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_23], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf248, primals_136, primals_135, buf249, 1024, 2560, grid=grid(1024), stream=stream0)
        buf253 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf254 = buf253; del buf253  # reuse
        buf255 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        buf257 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf258 = buf257; del buf257  # reuse
        buf259 = empty_strided_cuda((1, 1024, 2, 1), (2048, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_24, norm_8, denom_8, normalized_weight_8], Original ATen: [aten._weight_norm_interface, aten.linalg_vector_norm, aten.clamp_min, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_clamp_min_div_linalg_vector_norm_6.run(buf254, buf258, primals_139, primals_138, buf255, buf259, 1, 2048, grid=grid(1), stream=stream0)
        buf1 = empty_strided_cuda((4, 128, 64), (8192, 64, 1), torch.bool)
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_10.run(buf2, primals_2, buf1, 32768, grid=grid(32768), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(reinterpret_tensor(buf24, (4, 1, 33), (33, 33, 1), 0), primals_17, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf25, (4, 128, 33), (4224, 33, 1))
        buf49 = empty_strided_cuda((4, 1, 1, 17), (17, 17, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_11.run(buf24, buf49, 68, grid=grid(68), stream=stream0)
        buf100 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_63, primals_62, buf99, buf100, 160, grid=grid(160), stream=stream0)
        buf133 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_5], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_79, primals_78, buf132, buf133, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(reinterpret_tensor(buf131, (4, 1, 22, 3), (66, 66, 3, 1), 0), buf133, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 32, 8, 3), (768, 24, 3, 1))
        buf166 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_10], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_95, primals_94, buf165, buf166, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(reinterpret_tensor(buf164, (4, 1, 13, 5), (65, 0, 5, 1), 0), buf166, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 32, 5, 5), (800, 25, 5, 1))
        buf199 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_15], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_111, primals_110, buf198, buf199, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(reinterpret_tensor(buf197, (4, 1, 10, 7), (70, 0, 7, 1), 0), buf199, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 32, 4, 7), (896, 28, 7, 1))
        buf231 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_20], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_127, primals_126, buf230, buf231, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(reinterpret_tensor(buf131, (4, 1, 6, 11), (66, 66, 11, 1), 0), buf231, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 32, 2, 11), (704, 22, 11, 1))
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_4, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf3, (4, 128, 16), (2048, 16, 1))
        buf26 = empty_strided_cuda((4, 128, 33), (4224, 33, 1), torch.bool)
        buf27 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_13.run(buf27, primals_18, buf26, 16896, grid=grid(16896), stream=stream0)
        del primals_18
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(reinterpret_tensor(buf49, (4, 1, 17), (17, 17, 1), 0), primals_32, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf50, (4, 128, 17), (2176, 17, 1))
        buf74 = empty_strided_cuda((4, 1, 1, 9), (9, 9, 9, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_14.run(buf49, buf74, 36, grid=grid(36), stream=stream0)
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(reinterpret_tensor(primals_3, (4, 1, 32, 2), (64, 64, 2, 1), 0), buf100, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 32, 11, 2), (704, 22, 2, 1))
        buf135 = empty_strided_cuda((4, 32, 8, 3), (768, 24, 3, 1), torch.bool)
        buf136 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [input_62, input_63], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_15.run(buf136, primals_80, buf135, 3072, grid=grid(3072), stream=stream0)
        del primals_80
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf136, buf139, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 128, 3, 3), (1152, 9, 3, 1))
        buf168 = empty_strided_cuda((4, 32, 5, 5), (800, 25, 5, 1), torch.bool)
        buf169 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [input_71, input_72], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_16.run(buf169, primals_96, buf168, 3200, grid=grid(3200), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf169, buf172, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 128, 2, 5), (1280, 10, 5, 1))
        buf201 = empty_strided_cuda((4, 32, 4, 7), (896, 28, 7, 1), torch.bool)
        buf202 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_17.run(buf202, primals_112, buf201, 3584, grid=grid(3584), stream=stream0)
        del primals_112
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf202, buf205, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 128, 2, 7), (1792, 14, 7, 1))
        buf233 = empty_strided_cuda((4, 32, 2, 11), (704, 22, 11, 1), torch.bool)
        buf234 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [input_89, input_90], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf234, primals_128, buf233, 2816, grid=grid(2816), stream=stream0)
        del primals_128
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf234, buf237, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 128, 1, 11), (1408, 11, 11, 1))
        buf4 = empty_strided_cuda((4, 128, 16), (2048, 16, 1), torch.bool)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_19.run(buf5, primals_5, buf4, 8192, grid=grid(8192), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_19, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf28, (4, 128, 9), (1152, 9, 1))
        buf51 = empty_strided_cuda((4, 128, 17), (2176, 17, 1), torch.bool)
        buf52 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_20.run(buf52, primals_33, buf51, 8704, grid=grid(8704), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(reinterpret_tensor(buf74, (4, 1, 9), (9, 9, 1), 0), primals_47, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf75, (4, 128, 9), (1152, 9, 1))
        buf102 = empty_strided_cuda((4, 32, 11, 2), (704, 22, 2, 1), torch.bool)
        buf103 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [input_53, input_54], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf103, primals_64, buf102, 2816, grid=grid(2816), stream=stream0)
        del primals_64
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf103, buf106, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 128, 4, 2), (1024, 8, 2, 1))
        buf141 = empty_strided_cuda((4, 128, 3, 3), (1152, 9, 3, 1), torch.bool)
        buf142 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [input_64, input_65], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_21.run(buf142, primals_83, buf141, 4608, grid=grid(4608), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf142, buf145, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 512, 1, 3), (1536, 3, 3, 1))
        buf174 = empty_strided_cuda((4, 128, 2, 5), (1280, 10, 5, 1), torch.bool)
        buf175 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [input_73, input_74], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_22.run(buf175, primals_99, buf174, 5120, grid=grid(5120), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf175, buf178, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 512, 1, 5), (2560, 5, 5, 1))
        buf207 = empty_strided_cuda((4, 128, 2, 7), (1792, 14, 7, 1), torch.bool)
        buf208 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [input_82, input_83], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_23.run(buf208, primals_115, buf207, 7168, grid=grid(7168), stream=stream0)
        del primals_115
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf208, buf211, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 512, 1, 7), (3584, 7, 7, 1))
        buf239 = empty_strided_cuda((4, 128, 1, 11), (1408, 11, 11, 1), torch.bool)
        buf240 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [input_91, input_92], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_24.run(buf240, primals_131, buf239, 5632, grid=grid(5632), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf240, buf243, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 512, 1, 11), (5632, 11, 11, 1))
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_6, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf6, (4, 256, 4), (1024, 4, 1))
        buf29 = empty_strided_cuda((4, 128, 9), (1152, 9, 1), torch.bool)
        buf30 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_21.run(buf30, primals_20, buf29, 4608, grid=grid(4608), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_34, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf53, (4, 128, 5), (640, 5, 1))
        buf76 = empty_strided_cuda((4, 128, 9), (1152, 9, 1), torch.bool)
        buf77 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [input_40, input_41], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_21.run(buf77, primals_48, buf76, 4608, grid=grid(4608), stream=stream0)
        del primals_48
        buf108 = empty_strided_cuda((4, 128, 4, 2), (1024, 8, 2, 1), torch.bool)
        buf109 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [input_55, input_56], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_25.run(buf109, primals_67, buf108, 4096, grid=grid(4096), stream=stream0)
        del primals_67
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf109, buf112, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf147 = empty_strided_cuda((4, 512, 1, 3), (1536, 3, 3, 1), torch.bool)
        buf148 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [input_66, input_67], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_26.run(buf148, primals_86, buf147, 6144, grid=grid(6144), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf148, buf151, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 1024, 1, 3), (3072, 3, 3, 1))
        buf180 = empty_strided_cuda((4, 512, 1, 5), (2560, 5, 5, 1), torch.bool)
        buf181 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_75, input_76], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_27.run(buf181, primals_102, buf180, 10240, grid=grid(10240), stream=stream0)
        del primals_102
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf181, buf184, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 1024, 1, 5), (5120, 5, 5, 1))
        buf213 = empty_strided_cuda((4, 512, 1, 7), (3584, 7, 7, 1), torch.bool)
        buf214 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [input_84, input_85], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_28.run(buf214, primals_118, buf213, 14336, grid=grid(14336), stream=stream0)
        del primals_118
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf214, buf217, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 1024, 1, 7), (7168, 7, 7, 1))
        buf245 = empty_strided_cuda((4, 512, 1, 11), (5632, 11, 11, 1), torch.bool)
        buf246 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [input_93, input_94], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_29.run(buf246, primals_134, buf245, 22528, grid=grid(22528), stream=stream0)
        del primals_134
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf246, buf249, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 1024, 1, 11), (11264, 11, 11, 1))
        buf7 = empty_strided_cuda((4, 256, 4), (1024, 4, 1), torch.bool)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_30.run(buf8, primals_7, buf7, 4096, grid=grid(4096), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_21, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf31, (4, 256, 3), (768, 3, 1))
        buf54 = empty_strided_cuda((4, 128, 5), (640, 5, 1), torch.bool)
        buf55 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_31.run(buf55, primals_35, buf54, 2560, grid=grid(2560), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_49, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf78, (4, 128, 3), (384, 3, 1))
        buf114 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.bool)
        buf115 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_32.run(buf115, primals_70, buf114, 8192, grid=grid(8192), stream=stream0)
        del primals_70
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf115, buf118, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 1024, 2, 2), (4096, 4, 2, 1))
        buf153 = empty_strided_cuda((4, 1024, 1, 3), (3072, 3, 3, 1), torch.bool)
        buf154 = buf152; del buf152  # reuse
        buf158 = empty_strided_cuda((4, 1024, 1, 3), (3072, 3, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_68, input_69, input_70], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_33.run(buf154, primals_89, primals_92, buf153, buf158, 12288, grid=grid(12288), stream=stream0)
        del primals_89
        del primals_92
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf158, buf161, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 1, 2, 3), (6, 6, 3, 1))
        buf186 = empty_strided_cuda((4, 1024, 1, 5), (5120, 5, 5, 1), torch.bool)
        buf187 = buf185; del buf185  # reuse
        buf191 = empty_strided_cuda((4, 1024, 1, 5), (5120, 5, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_77, input_78, input_79], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_34.run(buf187, primals_105, primals_108, buf186, buf191, 20480, grid=grid(20480), stream=stream0)
        del primals_105
        del primals_108
        # Topologically Sorted Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf191, buf194, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 1, 2, 5), (10, 10, 5, 1))
        buf219 = empty_strided_cuda((4, 1024, 1, 7), (7168, 7, 7, 1), torch.bool)
        buf220 = buf218; del buf218  # reuse
        buf224 = empty_strided_cuda((4, 1024, 1, 7), (7168, 7, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_86, input_87, input_88], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_35.run(buf220, primals_121, primals_124, buf219, buf224, 28672, grid=grid(28672), stream=stream0)
        del primals_121
        del primals_124
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf224, buf227, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 1, 2, 7), (14, 14, 7, 1))
        buf251 = empty_strided_cuda((4, 1024, 1, 11), (11264, 11, 11, 1), torch.bool)
        buf252 = buf250; del buf250  # reuse
        buf256 = empty_strided_cuda((4, 1024, 1, 11), (11264, 11, 11, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_95, input_96, input_97], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_36.run(buf252, primals_137, primals_140, buf251, buf256, 45056, grid=grid(45056), stream=stream0)
        del primals_137
        del primals_140
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf256, buf259, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 1, 2, 11), (22, 22, 11, 1))
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_8, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf9, (4, 512, 1), (512, 1, 1))
        buf32 = empty_strided_cuda((4, 256, 3), (768, 3, 1), torch.bool)
        buf33 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_37.run(buf33, primals_22, buf32, 3072, grid=grid(3072), stream=stream0)
        del primals_22
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_36, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf56, (4, 256, 2), (512, 2, 1))
        buf79 = empty_strided_cuda((4, 128, 3), (384, 3, 1), torch.bool)
        buf80 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [input_42, input_43], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_38.run(buf80, primals_50, buf79, 1536, grid=grid(1536), stream=stream0)
        del primals_50
        buf120 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.bool)
        buf121 = buf119; del buf119  # reuse
        buf125 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_59, input_60, input_61], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_39.run(buf121, primals_73, primals_76, buf120, buf125, 16384, grid=grid(16384), stream=stream0)
        del primals_73
        del primals_76
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf125, buf128, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 1, 3, 2), (6, 6, 2, 1))
        buf163 = empty_strided_cuda((4, 1, 2, 3), (6, 6, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_40.run(buf162, primals_93, buf163, 24, grid=grid(24), stream=stream0)
        buf196 = empty_strided_cuda((4, 1, 2, 5), (10, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_41.run(buf195, primals_109, buf196, 40, grid=grid(40), stream=stream0)
        buf229 = empty_strided_cuda((4, 1, 2, 7), (14, 14, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_15], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_42.run(buf228, primals_125, buf229, 56, grid=grid(56), stream=stream0)
        buf261 = empty_strided_cuda((4, 1, 2, 11), (22, 22, 11, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_43.run(buf260, primals_141, buf261, 88, grid=grid(88), stream=stream0)
        buf10 = empty_strided_cuda((4, 512, 1), (512, 1, 1), torch.bool)
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_44.run(buf11, primals_9, buf10, 2048, grid=grid(2048), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_23, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf34, (4, 512, 1), (512, 1, 1))
        buf57 = empty_strided_cuda((4, 256, 2), (512, 2, 1), torch.bool)
        buf58 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_45.run(buf58, primals_37, buf57, 2048, grid=grid(2048), stream=stream0)
        del primals_37
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_51, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf81, (4, 256, 1), (256, 1, 1))
        buf130 = empty_strided_cuda((4, 1, 3, 2), (6, 6, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_40.run(buf129, primals_77, buf130, 24, grid=grid(24), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_10, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf12, (4, 1024, 1), (1024, 1, 1))
        buf35 = empty_strided_cuda((4, 512, 1), (512, 1, 1), torch.bool)
        buf36 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_44.run(buf36, primals_24, buf35, 2048, grid=grid(2048), stream=stream0)
        del primals_24
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_38, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf59, (4, 512, 1), (512, 1, 1))
        buf82 = empty_strided_cuda((4, 256, 1), (256, 1, 1), torch.bool)
        buf83 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_46.run(buf83, primals_52, buf82, 1024, grid=grid(1024), stream=stream0)
        del primals_52
        buf13 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf14 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_47.run(buf14, primals_11, buf13, 4096, grid=grid(4096), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_25, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf37, (4, 1024, 1), (1024, 1, 1))
        buf60 = empty_strided_cuda((4, 512, 1), (512, 1, 1), torch.bool)
        buf61 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_44.run(buf61, primals_39, buf60, 2048, grid=grid(2048), stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_53, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf84, (4, 512, 1), (512, 1, 1))
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_12, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf15, (4, 1024, 1), (1024, 1, 1))
        buf38 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf39 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_47.run(buf39, primals_26, buf38, 4096, grid=grid(4096), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_40, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf62, (4, 1024, 1), (1024, 1, 1))
        buf85 = empty_strided_cuda((4, 512, 1), (512, 1, 1), torch.bool)
        buf86 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_46, input_47], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_44.run(buf86, primals_54, buf85, 2048, grid=grid(2048), stream=stream0)
        del primals_54
        buf16 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf17 = buf15; del buf15  # reuse
        buf18 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_48.run(buf17, primals_13, primals_14, buf16, buf18, 4096, grid=grid(4096), stream=stream0)
        del primals_13
        del primals_14
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf18, buf21, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf22, (4, 1, 1), (1, 1, 1))
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_27, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf40, (4, 1024, 1), (1024, 1, 1))
        buf63 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf64 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_35, input_36], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_47.run(buf64, primals_41, buf63, 4096, grid=grid(4096), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_55, stride=(1,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf87, (4, 1024, 1), (1024, 1, 1))
        buf23 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_49.run(buf22, primals_16, buf23, 4, grid=grid(4), stream=stream0)
        buf41 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf42 = buf40; del buf40  # reuse
        buf43 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_48.run(buf42, primals_28, primals_29, buf41, buf43, 4096, grid=grid(4096), stream=stream0)
        del primals_28
        del primals_29
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf43, buf46, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf47, (4, 1, 1), (1, 1, 1))
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_42, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf65, (4, 1024, 1), (1024, 1, 1))
        buf88 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf89 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_47.run(buf89, primals_56, buf88, 4096, grid=grid(4096), stream=stream0)
        del primals_56
        buf48 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_49.run(buf47, primals_31, buf48, 4, grid=grid(4), stream=stream0)
        buf66 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf67 = buf65; del buf65  # reuse
        buf68 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_37, input_38, input_39], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_48.run(buf67, primals_43, primals_44, buf66, buf68, 4096, grid=grid(4096), stream=stream0)
        del primals_43
        del primals_44
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf68, buf71, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf72, (4, 1, 1), (1, 1, 1))
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_57, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf90, (4, 1024, 1), (1024, 1, 1))
        buf73 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_49.run(buf72, primals_46, buf73, 4, grid=grid(4), stream=stream0)
        buf91 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf92 = buf90; del buf90  # reuse
        buf93 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_50, input_51, input_52], Original ATen: [aten.convolution, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_leaky_relu_48.run(buf92, primals_58, primals_59, buf91, buf93, 4096, grid=grid(4096), stream=stream0)
        del primals_58
        del primals_59
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf93, buf96, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf97, (4, 1, 1), (1, 1, 1))
        buf98 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_49.run(buf97, primals_61, buf98, 4, grid=grid(4), stream=stream0)
    return (buf23, buf48, buf73, buf98, buf130, buf163, buf196, buf229, buf261, buf2, buf5, buf8, buf11, buf14, buf17, buf27, buf30, buf33, buf36, buf39, buf42, buf52, buf55, buf58, buf61, buf64, buf67, buf77, buf80, buf83, buf86, buf89, buf92, buf103, buf109, buf115, buf121, buf136, buf142, buf148, buf154, buf169, buf175, buf181, buf187, buf202, buf208, buf214, buf220, buf234, buf240, buf246, buf252, buf100, buf106, buf112, buf118, buf124, buf133, buf139, buf145, buf151, buf157, buf166, buf172, buf178, buf184, buf190, buf199, buf205, buf211, buf217, buf223, buf231, buf237, buf243, buf249, buf255, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_15, primals_16, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_30, primals_31, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_45, primals_46, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_68, primals_69, primals_71, primals_72, primals_74, primals_75, primals_77, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_123, primals_125, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf18, buf20, buf21, buf22, reinterpret_tensor(buf24, (4, 1, 33), (33, 33, 1), 0), buf26, buf27, buf29, buf30, buf32, buf33, buf35, buf36, buf38, buf39, buf41, buf43, buf45, buf46, buf47, reinterpret_tensor(buf49, (4, 1, 17), (17, 17, 1), 0), buf51, buf52, buf54, buf55, buf57, buf58, buf60, buf61, buf63, buf64, buf66, buf68, buf70, buf71, buf72, reinterpret_tensor(buf74, (4, 1, 9), (9, 9, 1), 0), buf76, buf77, buf79, buf80, buf82, buf83, buf85, buf86, buf88, buf89, buf91, buf93, buf95, buf96, buf97, buf99, buf100, buf102, buf103, buf105, buf106, buf108, buf109, buf111, buf112, buf114, buf115, buf117, buf118, buf120, buf123, buf124, buf125, buf127, buf128, buf129, buf131, buf132, buf133, buf135, buf136, buf138, buf139, buf141, buf142, buf144, buf145, buf147, buf148, buf150, buf151, buf153, buf156, buf157, buf158, buf160, buf161, buf162, reinterpret_tensor(buf164, (4, 1, 13, 5), (65, 65, 5, 1), 0), buf165, buf166, buf168, buf169, buf171, buf172, buf174, buf175, buf177, buf178, buf180, buf181, buf183, buf184, buf186, buf189, buf190, buf191, buf193, buf194, buf195, reinterpret_tensor(buf197, (4, 1, 10, 7), (70, 70, 7, 1), 0), buf198, buf199, buf201, buf202, buf204, buf205, buf207, buf208, buf210, buf211, buf213, buf214, buf216, buf217, buf219, buf222, buf223, buf224, buf226, buf227, buf228, buf230, buf231, buf233, buf234, buf236, buf237, buf239, buf240, buf242, buf243, buf245, buf246, buf248, buf249, buf251, buf254, buf255, buf256, buf258, buf259, buf260, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 1, 15), (15, 15, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, 8, 41), (328, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, 16, 41), (656, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1024, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, 1024, 5), (5120, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1, 1024, 3), (3072, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 1, 15), (15, 15, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, 8, 41), (328, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, 16, 41), (656, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1024, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1024, 1024, 5), (5120, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, 1024, 3), (3072, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 1, 15), (15, 15, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, 8, 41), (328, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, 16, 41), (656, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1024, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1024, 1024, 5), (5120, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1, 1024, 3), (3072, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, 1, 15), (15, 15, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, 8, 41), (328, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, 16, 41), (656, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, 32, 41), (1312, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1024, 1024, 5), (5120, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1, 1024, 3), (3072, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1, 1024, 2, 1), (2048, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1, 1024, 2, 1), (2048, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1, 1024, 2, 1), (2048, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1, 1024, 2, 1), (2048, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1, 1024, 2, 1), (2048, 2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
