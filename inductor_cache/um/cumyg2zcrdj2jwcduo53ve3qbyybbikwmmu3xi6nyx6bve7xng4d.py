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


# kernel path: inductor_cache/gp/cgpstdojl5wp3oosriexw4hgfxyomfdvppo2b5wd5ghnwsymqghh.py
# Topologically Sorted Source Nodes: [add, reciprocal, mul, sin, pow_1, mul_1, x_1], Original ATen: [aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add => add
#   mul => mul
#   mul_1 => mul_1
#   pow_1 => pow_1
#   reciprocal => reciprocal
#   sin => sin
#   x_1 => add_1
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, 1e-09), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %view), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul,), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin, 2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, %pow_1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view, %mul_1), kwargs = {})
triton_poi_fused_add_mul_pow_reciprocal_sin_0 = async_compile.triton('triton_poi_fused_add_mul_pow_reciprocal_sin_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_reciprocal_sin_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_reciprocal_sin_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 8)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 1e-09
    tmp3 = tmp1 + tmp2
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp4 / tmp3
    tmp6 = tmp1 * tmp0
    tmp7 = tl_math.sin(tmp6)
    tmp8 = tmp7 * tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nt/cntyjx5zwo3fcrjo3ma5r2ita36hegb4abse2rw2pmsqhi7qjxdq.py
# Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm => div, mul_2, pow_2, pow_3, sum_1
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_4, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [1, 2], True), kwargs = {})
#   %pow_3 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_3, %pow_3), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, %div), kwargs = {})
triton_per_fused__weight_norm_interface_1 = async_compile.triton('triton_per_fused__weight_norm_interface_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 56
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 56*x0), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 56*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/az/cazozyvry3mxubdtvg6pjlqchtuqczruevudipakutnbwbbjmv64.py
# Topologically Sorted Source Nodes: [input_1, add_2, reciprocal_1, mul_2, sin_1, pow_2, mul_3, x_4], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_2 => add_2
#   input_1 => convolution
#   mul_2 => mul_3
#   mul_3 => mul_4
#   pow_2 => pow_4
#   reciprocal_1 => reciprocal_1
#   sin_1 => sin_1
#   x_4 => add_3
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1, %mul_2, %primals_5, [1], [3], [1], False, [0], 1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_6, 1e-09), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_2,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, %view_2), kwargs = {})
#   %sin_1 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_3,), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_1, 2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, %pow_4), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %mul_4), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 8)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1e-09
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp6 / tmp5
    tmp8 = tmp3 * tmp2
    tmp9 = tl_math.sin(tmp8)
    tmp10 = tmp9 * tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp2 + tmp11
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f6/cf6wtwitrskcuuissbxxin6ngneqnr3jr4twpdtnatayvy5b4nkb.py
# Topologically Sorted Source Nodes: [_weight_norm_1], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_1 => div_1, mul_5, pow_5, pow_6, sum_2
# Graph fragment:
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_8, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1, 2], True), kwargs = {})
#   %pow_6 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_7, %pow_6), kwargs = {})
#   %mul_5 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_8, %div_1), kwargs = {})
triton_per_fused__weight_norm_interface_3 = async_compile.triton('triton_per_fused__weight_norm_interface_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 8*x0), xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 8*x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mq/cmqs7f2lbewwodjobejzreqpdaosf7lwv2g3me2kkzj4w7zvsgpi.py
# Topologically Sorted Source Nodes: [input_2, input_3, x_6, add_5, reciprocal_2, mul_4, sin_2, pow_3, mul_5, x_7], Original ATen: [aten.convolution, aten.add, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_5 => add_5
#   input_2 => convolution_1
#   input_3 => add_4
#   mul_4 => mul_6
#   mul_5 => mul_7
#   pow_3 => pow_7
#   reciprocal_2 => reciprocal_2
#   sin_2 => sin_2
#   x_6 => view_4
#   x_7 => add_6
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_3, %mul_5, %primals_9, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %convolution_1), kwargs = {})
#   %view_4 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_4, [4, 8, -1]), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_10, 1e-09), kwargs = {})
#   %reciprocal_2 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_5,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_10, %view_4), kwargs = {})
#   %sin_2 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_6,), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_2, 2), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, %pow_7), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %mul_7), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_4 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 8)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tmp6 = 1e-09
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = tmp5 * tmp4
    tmp11 = tl_math.sin(tmp10)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp9 * tmp12
    tmp14 = tmp4 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ak/cakpa3t7tpcptq2isdlfehcsqzkusgqlsrpllbr2qnkk77j4ghe2.py
# Topologically Sorted Source Nodes: [input_3, input_5, input_6, x_12, add_10, reciprocal_4, mul_8, sin_4, pow_5, mul_9, x_13], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_10 => add_10
#   input_3 => add_4
#   input_5 => convolution_3
#   input_6 => add_9
#   mul_8 => mul_12
#   mul_9 => mul_13
#   pow_5 => pow_13
#   reciprocal_4 => reciprocal_4
#   sin_4 => sin_4
#   x_12 => view_8
#   x_13 => add_11
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %convolution_1), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %mul_11, %primals_17, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_3), kwargs = {})
#   %view_8 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_9, [4, 8, -1]), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_18, 1e-09), kwargs = {})
#   %reciprocal_4 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_10,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, %view_8), kwargs = {})
#   %sin_4 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_12,), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_4, 2), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, %pow_13), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %mul_13), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_5 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 8)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = 1e-09
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp10 / tmp9
    tmp12 = tmp7 * tmp6
    tmp13 = tl_math.sin(tmp12)
    tmp14 = tmp13 * tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp6 + tmp15
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4r/c4rlf5cyg73d73x6vnw2eewtmtz4qvplns2vupd7txgfc4hifklu.py
# Topologically Sorted Source Nodes: [input_3, input_5, input_6, input_8, input_9, x_18, add_15, reciprocal_6, mul_12, sin_6, pow_7, mul_13, x_19], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_15 => add_15
#   input_3 => add_4
#   input_5 => convolution_3
#   input_6 => add_9
#   input_8 => convolution_5
#   input_9 => add_14
#   mul_12 => mul_18
#   mul_13 => mul_19
#   pow_7 => pow_19
#   reciprocal_6 => reciprocal_6
#   sin_6 => sin_6
#   x_18 => view_12
#   x_19 => add_16
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %convolution_1), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %mul_11, %primals_17, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_3), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_13, %mul_17, %primals_25, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convolution_5), kwargs = {})
#   %view_12 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_14, [4, 8, -1]), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_26, 1e-09), kwargs = {})
#   %reciprocal_6 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_15,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_26, %view_12), kwargs = {})
#   %sin_6 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_18,), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_6, 2), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, %pow_19), kwargs = {})
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_12, %mul_19), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_6 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 8)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = 1e-09
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp14 / tmp13
    tmp16 = tmp11 * tmp10
    tmp17 = tl_math.sin(tmp16)
    tmp18 = tmp17 * tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = tmp10 + tmp19
    tl.store(in_out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hh/chhp66pine5oqpkrxicbyfmmjf7ew25z5t7vjw6kraqqqvjrqeyy.py
# Topologically Sorted Source Nodes: [_weight_norm_6], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_6 => div_6, mul_20, pow_20, pow_21, sum_7
# Graph fragment:
#   %pow_20 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_28, 2), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_20, [1, 2], True), kwargs = {})
#   %pow_21 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_7, 0.5), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_27, %pow_21), kwargs = {})
#   %mul_20 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_28, %div_6), kwargs = {})
triton_per_fused__weight_norm_interface_7 = async_compile.triton('triton_per_fused__weight_norm_interface_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_7(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 16*x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t6/ct6ul4xmfgs6mdvnx3uym34ccolisymnhh2diisu35dufts2wmcv.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_10 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_16, %mul_20, %primals_29, [1], [1], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_8 = async_compile.triton('triton_poi_fused_convolution_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 5) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29 = args
    args.clear()
    assert_size_stride(primals_1, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_2, (4, 8, 4), (32, 4, 1))
    assert_size_stride(primals_3, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_4, (8, 8, 7), (56, 7, 1))
    assert_size_stride(primals_5, (8, ), (1, ))
    assert_size_stride(primals_6, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_7, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_8, (8, 8, 1), (8, 1, 1))
    assert_size_stride(primals_9, (8, ), (1, ))
    assert_size_stride(primals_10, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_11, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_12, (8, 8, 7), (56, 7, 1))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_15, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_16, (8, 8, 1), (8, 1, 1))
    assert_size_stride(primals_17, (8, ), (1, ))
    assert_size_stride(primals_18, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_19, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_20, (8, 8, 7), (56, 7, 1))
    assert_size_stride(primals_21, (8, ), (1, ))
    assert_size_stride(primals_22, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_23, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_24, (8, 8, 1), (8, 1, 1))
    assert_size_stride(primals_25, (8, ), (1, ))
    assert_size_stride(primals_26, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_27, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_28, (16, 8, 2), (16, 2, 1))
    assert_size_stride(primals_29, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, reciprocal, mul, sin, pow_1, mul_1, x_1], Original ATen: [aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_reciprocal_sin_0.run(primals_2, primals_1, buf0, 128, grid=grid(128), stream=stream0)
        buf1 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf2 = reinterpret_tensor(buf1, (8, 1, 1), (1, 1, 1), 0); del buf1  # reuse
        buf3 = empty_strided_cuda((8, 8, 7), (56, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_1.run(buf2, primals_4, primals_3, buf3, 8, 56, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf0, buf3, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf4, (4, 8, 4), (32, 4, 1))
        buf5 = buf4; del buf4  # reuse
        buf6 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, add_2, reciprocal_1, mul_2, sin_1, pow_2, mul_3, x_4], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2.run(buf5, primals_5, primals_6, buf6, 128, grid=grid(128), stream=stream0)
        del primals_5
        buf7 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf8 = reinterpret_tensor(buf7, (8, 1, 1), (1, 1, 1), 0); del buf7  # reuse
        buf9 = empty_strided_cuda((8, 8, 1), (8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_1], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf8, primals_8, primals_7, buf9, 8, 8, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf6, buf9, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf10, (4, 8, 4), (32, 4, 1))
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3, x_6, add_5, reciprocal_2, mul_4, sin_2, pow_3, mul_5, x_7], Original ATen: [aten.convolution, aten.add, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_4.run(buf11, primals_9, primals_2, primals_10, buf12, 128, grid=grid(128), stream=stream0)
        del primals_9
        buf13 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf14 = reinterpret_tensor(buf13, (8, 1, 1), (1, 1, 1), 0); del buf13  # reuse
        buf15 = empty_strided_cuda((8, 8, 7), (56, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_2], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_1.run(buf14, primals_12, primals_11, buf15, 8, 56, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf12, buf15, stride=(1,), padding=(9,), dilation=(3,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf16, (4, 8, 4), (32, 4, 1))
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, add_7, reciprocal_3, mul_6, sin_3, pow_4, mul_7, x_10], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2.run(buf17, primals_13, primals_14, buf18, 128, grid=grid(128), stream=stream0)
        del primals_13
        buf19 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf20 = reinterpret_tensor(buf19, (8, 1, 1), (1, 1, 1), 0); del buf19  # reuse
        buf21 = empty_strided_cuda((8, 8, 1), (8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_3], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf20, primals_16, primals_15, buf21, 8, 8, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf18, buf21, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf22, (4, 8, 4), (32, 4, 1))
        buf23 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        buf24 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_5, input_6, x_12, add_10, reciprocal_4, mul_8, sin_4, pow_5, mul_9, x_13], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_5.run(primals_2, buf11, buf22, primals_17, primals_18, buf23, buf24, 128, grid=grid(128), stream=stream0)
        buf25 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf26 = reinterpret_tensor(buf25, (8, 1, 1), (1, 1, 1), 0); del buf25  # reuse
        buf27 = empty_strided_cuda((8, 8, 7), (56, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_4], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_1.run(buf26, primals_20, primals_19, buf27, 8, 56, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf24, buf27, stride=(1,), padding=(27,), dilation=(9,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf28, (4, 8, 4), (32, 4, 1))
        buf29 = buf28; del buf28  # reuse
        buf30 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, add_12, reciprocal_5, mul_10, sin_5, pow_6, mul_11, x_16], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2.run(buf29, primals_21, primals_22, buf30, 128, grid=grid(128), stream=stream0)
        del primals_21
        buf31 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf32 = reinterpret_tensor(buf31, (8, 1, 1), (1, 1, 1), 0); del buf31  # reuse
        buf33 = empty_strided_cuda((8, 8, 1), (8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_5], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf32, primals_24, primals_23, buf33, 8, 8, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf30, buf33, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf34, (4, 8, 4), (32, 4, 1))
        buf35 = buf22; del buf22  # reuse
        buf36 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_5, input_6, input_8, input_9, x_18, add_15, reciprocal_6, mul_12, sin_6, pow_7, mul_13, x_19], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_6.run(buf35, primals_2, buf11, primals_17, buf34, primals_25, primals_26, buf36, 128, grid=grid(128), stream=stream0)
        del buf34
        del primals_17
        del primals_25
        buf37 = empty_strided_cuda((16, 1, 1), (1, 16, 16), torch.float32)
        buf38 = reinterpret_tensor(buf37, (16, 1, 1), (1, 1, 1), 0); del buf37  # reuse
        buf39 = empty_strided_cuda((16, 8, 2), (16, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_6], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_7.run(buf38, primals_28, primals_27, buf39, 16, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf36, buf39, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf40, (4, 16, 5), (80, 5, 1))
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_8.run(buf41, primals_29, 320, grid=grid(320), stream=stream0)
        del primals_29
    return (buf41, buf3, buf9, buf15, buf21, buf27, buf33, buf39, primals_1, primals_2, primals_3, primals_4, primals_6, primals_7, primals_8, primals_10, primals_11, primals_12, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, buf0, buf2, buf3, buf5, buf6, buf8, buf9, buf11, buf12, buf14, buf15, buf17, buf18, buf20, buf21, buf23, buf24, buf26, buf27, buf29, buf30, buf32, buf33, buf35, buf36, buf38, buf39, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 8, 4), (32, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((8, 8, 7), (56, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, 8, 7), (56, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((8, 8, 7), (56, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((8, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((16, 8, 2), (16, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
