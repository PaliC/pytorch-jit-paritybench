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


# kernel path: inductor_cache/dh/cdhokkc32rvbuhsye57oy2pjuuqsvncmdw5iojyjdtcelhznd5p2.py
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
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_reciprocal_sin_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_reciprocal_sin_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 16)
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


# kernel path: inductor_cache/g6/cg6vasv5733d6ryol7shjsn3y3us56ck3bnc7rpnblapfhx3ytpb.py
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
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ww/cww7xtki5q7marobvhgthsi6jc36vrlmbhlatj57d57csiwdyheo.py
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
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1, %mul_2, %primals_5, [1], [1], [1], True, [0], 1), kwargs = {})
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
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 8)
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


# kernel path: inductor_cache/4c/c4c7lcsj4nm63zp2ctvhlofmdaulg4o5a6o536zflp37sizjy42x.py
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
    size_hints={'x': 8, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/2y/c2yocb5ov62ab7sx6grok4cgexs2hdocgeuz3bvqvqbjz4o5cvaz.py
# Topologically Sorted Source Nodes: [_weight_norm_2], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_2 => div_2, mul_8, pow_8, pow_9, sum_3
# Graph fragment:
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_12, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_8, [1, 2], True), kwargs = {})
#   %pow_9 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_11, %pow_9), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, %div_2), kwargs = {})
triton_per_fused__weight_norm_interface_4 = async_compile.triton('triton_per_fused__weight_norm_interface_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/mp/cmpaarla4uhklekq2daevupoexqciec7du62wiqr77o733v5iykf.py
# Topologically Sorted Source Nodes: [input_3, input_4, x_9, add_7, reciprocal_3, mul_6, sin_3, pow_4, mul_7, x_10], Original ATen: [aten.convolution, aten.add, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_7 => add_7
#   input_3 => convolution_2
#   input_4 => add_6
#   mul_6 => mul_9
#   mul_7 => mul_10
#   pow_4 => pow_10
#   reciprocal_3 => reciprocal_3
#   sin_3 => sin_3
#   x_10 => add_8
#   x_9 => view_6
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_5, %mul_8, %primals_13, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %view_6 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [4, 8, -1]), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_14, 1e-09), kwargs = {})
#   %reciprocal_3 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_7,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_14, %view_6), kwargs = {})
#   %sin_3 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_9,), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_3, 2), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, %pow_10), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %mul_10), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 8)
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


# kernel path: inductor_cache/h4/ch47vha3qz2hpgteoey72st6d5mynf6jwmcfyxa3ycvxoepqbcqq.py
# Topologically Sorted Source Nodes: [input_4, input_6, input_7, x_15, add_12, reciprocal_5, mul_10, sin_5, pow_6, mul_11, x_16], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_12 => add_12
#   input_4 => add_6
#   input_6 => convolution_4
#   input_7 => add_11
#   mul_10 => mul_15
#   mul_11 => mul_16
#   pow_6 => pow_16
#   reciprocal_5 => reciprocal_5
#   sin_5 => sin_5
#   x_15 => view_10
#   x_16 => add_13
# Graph fragment:
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_10, %mul_14, %primals_21, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %convolution_4), kwargs = {})
#   %view_10 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_11, [4, 8, -1]), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_22, 1e-09), kwargs = {})
#   %reciprocal_5 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_12,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_22, %view_10), kwargs = {})
#   %sin_5 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_15,), kwargs = {})
#   %pow_16 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_5, 2), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, %pow_16), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_10, %mul_16), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 8)
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


# kernel path: inductor_cache/j4/cj42tsl37g5w7tivcxc7xctf3dcdausvzu74mamylblvyrx7lwqn.py
# Topologically Sorted Source Nodes: [input_4, input_6, input_7, input_9, input_10], Original ATen: [aten.add, aten.convolution]
# Source node to ATen node mapping:
#   input_10 => add_16
#   input_4 => add_6
#   input_6 => convolution_4
#   input_7 => add_11
#   input_9 => convolution_6
# Graph fragment:
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_10, %mul_14, %primals_21, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %convolution_4), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_15, %mul_20, %primals_29, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %convolution_6), kwargs = {})
triton_poi_fused_add_convolution_7 = async_compile.triton('triton_poi_fused_add_convolution_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 8)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29 = args
    args.clear()
    assert_size_stride(primals_1, (1, 16, 1), (16, 1, 1))
    assert_size_stride(primals_2, (4, 16, 4), (64, 4, 1))
    assert_size_stride(primals_3, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_4, (16, 8, 2), (16, 2, 1))
    assert_size_stride(primals_5, (8, ), (1, ))
    assert_size_stride(primals_6, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_7, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_8, (8, 8, 7), (56, 7, 1))
    assert_size_stride(primals_9, (8, ), (1, ))
    assert_size_stride(primals_10, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_11, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_12, (8, 8, 1), (8, 1, 1))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_15, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_16, (8, 8, 7), (56, 7, 1))
    assert_size_stride(primals_17, (8, ), (1, ))
    assert_size_stride(primals_18, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_19, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_20, (8, 8, 1), (8, 1, 1))
    assert_size_stride(primals_21, (8, ), (1, ))
    assert_size_stride(primals_22, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_23, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_24, (8, 8, 7), (56, 7, 1))
    assert_size_stride(primals_25, (8, ), (1, ))
    assert_size_stride(primals_26, (1, 8, 1), (8, 1, 1))
    assert_size_stride(primals_27, (8, 1, 1), (1, 1, 1))
    assert_size_stride(primals_28, (8, 8, 1), (8, 1, 1))
    assert_size_stride(primals_29, (8, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, reciprocal, mul, sin, pow_1, mul_1, x_1], Original ATen: [aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_reciprocal_sin_0.run(primals_2, primals_1, buf0, 256, grid=grid(256), stream=stream0)
        buf1 = empty_strided_cuda((16, 1, 1), (1, 16, 16), torch.float32)
        buf2 = reinterpret_tensor(buf1, (16, 1, 1), (1, 1, 1), 0); del buf1  # reuse
        buf3 = empty_strided_cuda((16, 8, 2), (16, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_1.run(buf2, primals_4, primals_3, buf3, 16, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf0, buf3, stride=(1,), padding=(1,), dilation=(1,), transposed=True, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf4, (4, 8, 3), (24, 3, 1))
        buf5 = buf4; del buf4  # reuse
        buf6 = empty_strided_cuda((4, 8, 3), (24, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, add_2, reciprocal_1, mul_2, sin_1, pow_2, mul_3, x_4], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2.run(buf5, primals_5, primals_6, buf6, 96, grid=grid(96), stream=stream0)
        del primals_5
        buf7 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf8 = reinterpret_tensor(buf7, (8, 1, 1), (1, 1, 1), 0); del buf7  # reuse
        buf9 = empty_strided_cuda((8, 8, 7), (56, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_1], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf8, primals_8, primals_7, buf9, 8, 56, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf6, buf9, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf10, (4, 8, 3), (24, 3, 1))
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((4, 8, 3), (24, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, add_4, reciprocal_2, mul_4, sin_2, pow_3, mul_5, x_7], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2.run(buf11, primals_9, primals_10, buf12, 96, grid=grid(96), stream=stream0)
        del primals_9
        buf13 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf14 = reinterpret_tensor(buf13, (8, 1, 1), (1, 1, 1), 0); del buf13  # reuse
        buf15 = empty_strided_cuda((8, 8, 1), (8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_2], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf14, primals_12, primals_11, buf15, 8, 8, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf12, buf15, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf16, (4, 8, 3), (24, 3, 1))
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((4, 8, 3), (24, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4, x_9, add_7, reciprocal_3, mul_6, sin_3, pow_4, mul_7, x_10], Original ATen: [aten.convolution, aten.add, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_5.run(buf17, primals_13, buf5, primals_14, buf18, 96, grid=grid(96), stream=stream0)
        del primals_13
        buf19 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf20 = reinterpret_tensor(buf19, (8, 1, 1), (1, 1, 1), 0); del buf19  # reuse
        buf21 = empty_strided_cuda((8, 8, 7), (56, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_3], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf20, primals_16, primals_15, buf21, 8, 56, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf18, buf21, stride=(1,), padding=(9,), dilation=(3,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf22, (4, 8, 3), (24, 3, 1))
        buf23 = buf22; del buf22  # reuse
        buf24 = empty_strided_cuda((4, 8, 3), (24, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, add_9, reciprocal_4, mul_8, sin_4, pow_5, mul_9, x_13], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2.run(buf23, primals_17, primals_18, buf24, 96, grid=grid(96), stream=stream0)
        del primals_17
        buf25 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf26 = reinterpret_tensor(buf25, (8, 1, 1), (1, 1, 1), 0); del buf25  # reuse
        buf27 = empty_strided_cuda((8, 8, 1), (8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_4], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf26, primals_20, primals_19, buf27, 8, 8, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf24, buf27, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf28, (4, 8, 3), (24, 3, 1))
        buf29 = empty_strided_cuda((4, 8, 3), (24, 3, 1), torch.float32)
        buf30 = empty_strided_cuda((4, 8, 3), (24, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_6, input_7, x_15, add_12, reciprocal_5, mul_10, sin_5, pow_6, mul_11, x_16], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_6.run(buf5, buf17, buf28, primals_21, primals_22, buf29, buf30, 96, grid=grid(96), stream=stream0)
        buf31 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf32 = reinterpret_tensor(buf31, (8, 1, 1), (1, 1, 1), 0); del buf31  # reuse
        buf33 = empty_strided_cuda((8, 8, 7), (56, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_5], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf32, primals_24, primals_23, buf33, 8, 56, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf30, buf33, stride=(1,), padding=(27,), dilation=(9,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf34, (4, 8, 3), (24, 3, 1))
        buf35 = buf34; del buf34  # reuse
        buf36 = empty_strided_cuda((4, 8, 3), (24, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, add_14, reciprocal_6, mul_12, sin_6, pow_7, mul_13, x_19], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_2.run(buf35, primals_25, primals_26, buf36, 96, grid=grid(96), stream=stream0)
        del primals_25
        buf37 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
        buf38 = reinterpret_tensor(buf37, (8, 1, 1), (1, 1, 1), 0); del buf37  # reuse
        buf39 = empty_strided_cuda((8, 8, 1), (8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_6], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf38, primals_28, primals_27, buf39, 8, 8, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf36, buf39, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf40, (4, 8, 3), (24, 3, 1))
        buf41 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_6, input_7, input_9, input_10], Original ATen: [aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_7.run(buf41, buf5, buf17, primals_21, buf40, primals_29, 96, grid=grid(96), stream=stream0)
        del buf40
        del primals_21
        del primals_29
    return (buf41, buf3, buf9, buf15, buf21, buf27, buf33, buf39, primals_1, primals_2, primals_3, primals_4, primals_6, primals_7, primals_8, primals_10, primals_11, primals_12, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, buf0, buf2, buf3, buf5, buf6, buf8, buf9, buf11, buf12, buf14, buf15, buf17, buf18, buf20, buf21, buf23, buf24, buf26, buf27, buf29, buf30, buf32, buf33, buf35, buf36, buf38, buf39, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 16, 4), (64, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 8, 2), (16, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, 8, 7), (56, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, 8, 7), (56, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((8, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((8, 8, 7), (56, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((8, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
