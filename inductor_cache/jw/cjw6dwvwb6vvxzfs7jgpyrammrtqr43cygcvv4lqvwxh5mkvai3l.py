# AOT ID: ['31_forward']
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


# kernel path: inductor_cache/3p/c3pfq4df46jk3zzan3t6k4rjx4v7g6jj6pwff6pt2dvyzgvs5rg7.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_1 => add, clone, rsqrt, var_mean
# Graph fragment:
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 4), kwargs = {})
triton_poi_fused_native_layer_norm_native_layer_norm_backward_0 = async_compile.triton('triton_poi_fused_native_layer_norm_native_layer_norm_backward_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_native_layer_norm_backward_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_native_layer_norm_backward_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr1 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp10 = tl.load(in_ptr1 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp15 = tl.load(in_ptr1 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp7 = tmp4 + tmp6
    tmp8 = tmp3 + tmp7
    tmp12 = tmp9 + tmp11
    tmp13 = tmp8 + tmp12
    tmp17 = tmp14 + tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = 4.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp3 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tmp7 - tmp20
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 + tmp24
    tmp26 = tmp12 - tmp20
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 + tmp27
    tmp29 = tmp17 - tmp20
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 + tmp30
    tmp32 = tmp31 / tmp19
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = 0.25
    tmp37 = tmp35 * tmp36
    tl.store(out_ptr0 + (x2), tmp20, xmask)
    tl.store(out_ptr1 + (x2), tmp32, xmask)
    tl.store(out_ptr2 + (x2), tmp37, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vf/cvfn6e7n2z6bl7lnm2epmycdprnz4ffjgvis6wqk2yd7n2m5scll.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_1 => add, clone, mul, rsqrt, sub
# Graph fragment:
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem_1), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
triton_poi_fused_native_layer_norm_1 = async_compile.triton('triton_poi_fused_native_layer_norm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x2 + 64*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.store(out_ptr0 + (x2 + 4*y3), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/mk/cmk7rwlkwy2ukinbkuo76hxzppq4vh7hi3zrg5hv2wo5ozgal4eu.py
# Topologically Sorted Source Nodes: [input_2, agg_norm], Original ATen: [aten.gelu, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   agg_norm => pow_1, pow_2, sum_1
#   input_2 => add_1, erf, mul_1, mul_2, mul_3
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.5), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_2,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %add_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_3, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   %pow_2 : [num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
triton_per_fused_gelu_linalg_vector_norm_2 = async_compile.triton('triton_per_fused_gelu_linalg_vector_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_linalg_vector_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_gelu_linalg_vector_norm_2(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 16)
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*r2 + 256*x1), xmask, other=0.0)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = libdevice.sqrt(tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/st/cstimtfy74cb2nqlik2q53dpfg2v4w6vjcohgrw3kmzuczzwrasi.py
# Topologically Sorted Source Nodes: [mean, add], Original ATen: [aten.mean, aten.add]
# Source node to ATen node mapping:
#   add => add_2
#   mean => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
triton_per_fused_add_mean_3 = async_compile.triton('triton_per_fused_add_mean_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_3(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/iv/civlcxuh3txjclv6furr4nng5lz2y36zqodyho6didzxi4ay33gt.py
# Topologically Sorted Source Nodes: [input_2, stand_div_norm, mul, mul_1, add_1, input_3], Original ATen: [aten.gelu, aten.div, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_3
#   input_2 => add_1, erf, mul_1, mul_2, mul_3
#   input_3 => add_4
#   mul => mul_4
#   mul_1 => mul_5
#   stand_div_norm => div
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.5), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_2,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %add_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_2, %add_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %div), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, %mul_4), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %primals_7), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_3), kwargs = {})
triton_poi_fused_add_div_gelu_mul_4 = async_compile.triton('triton_poi_fused_add_div_gelu_mul_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_gelu_mul_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_gelu_mul_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x3 = xindex
    x2 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp10 = tl.load(in_ptr2 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp12 = tmp10 / tmp11
    tmp13 = tmp9 * tmp12
    tmp14 = tmp0 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 + tmp9
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wy/cwycoh6leq7cxltlk4gqfyergojbbsdhiujc2cauimyrvvj4p7cr.py
# Topologically Sorted Source Nodes: [add_3], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_3 => add_5
# Graph fragment:
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_5, %primals_1), kwargs = {})
triton_poi_fused_add_5 = async_compile.triton('triton_poi_fused_add_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_5(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 16*x2 + 64*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 4*y3), tmp4, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (16, 4), (4, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (1, 1, 1, 16), (16, 16, 16, 1))
    assert_size_stride(primals_7, (1, 1, 1, 16), (16, 16, 16, 1))
    assert_size_stride(primals_8, (4, 16), (16, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf0, (4, 4, 4, 4), (64, 16, 4, 1))
        buf1 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        buf2 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        buf12 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_native_layer_norm_backward_0.run(buf0, primals_3, buf1, buf2, buf12, 64, grid=grid(64), stream=stream0)
        buf3 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_1.run(buf0, primals_3, buf1, buf2, buf3, 64, 4, grid=grid(64, 4), stream=stream0)
        del buf1
        del primals_3
        buf4 = empty_strided_cuda((64, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(buf3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 16), (1, 4), 0), alpha=1, beta=1, out=buf4)
        del primals_5
        buf5 = reinterpret_tensor(buf2, (4, 1, 1, 16), (16, 64, 64, 1), 0); del buf2  # reuse
        buf6 = reinterpret_tensor(buf5, (4, 1, 1, 16), (16, 16, 16, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_2, agg_norm], Original ATen: [aten.gelu, aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_gelu_linalg_vector_norm_2.run(buf6, buf4, 64, 16, grid=grid(64), stream=stream0)
        buf7 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf8 = reinterpret_tensor(buf7, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [mean, add], Original ATen: [aten.mean, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_3.run(buf8, buf6, 4, 16, grid=grid(4), stream=stream0)
        buf9 = empty_strided_cuda((4, 4, 4, 16), (256, 64, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, stand_div_norm, mul, mul_1, add_1, input_3], Original ATen: [aten.gelu, aten.div, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_gelu_mul_4.run(primals_6, buf4, buf6, buf8, primals_7, buf9, 1024, grid=grid(1024), stream=stream0)
        del primals_7
        buf10 = reinterpret_tensor(buf0, (64, 4), (4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf9, (64, 16), (16, 1), 0), reinterpret_tensor(primals_8, (16, 4), (1, 16), 0), out=buf10)
        buf11 = reinterpret_tensor(buf10, (4, 4, 4, 4), (64, 1, 16, 4), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [add_3], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_5.run(buf11, primals_9, primals_1, 64, 4, grid=grid(64, 4), stream=stream0)
        del primals_9
    return (buf11, primals_1, primals_2, primals_6, buf3, buf4, buf6, buf8, reinterpret_tensor(buf9, (64, 16), (16, 1), 0), primals_8, primals_4, buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 1, 1, 16), (16, 16, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, 1, 1, 16), (16, 16, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
