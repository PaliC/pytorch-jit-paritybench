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


# kernel path: inductor_cache/nt/cntntp5re3nuarpvdavuh6axtc7vyfr5qwnyrv5x6fcumuj6pqqc.py
# Topologically Sorted Source Nodes: [x, mul, x_att], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   mul => mul
#   x => add
#   x_att => var_mean
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %slice_2), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %primals_3), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul, [2]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused_add_mul_native_layer_norm_0 = async_compile.triton('triton_poi_fused_add_mul_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 * tmp20
    tmp22 = tmp16 + tmp21
    tmp23 = 4.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp4 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tmp9 - tmp24
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 + tmp28
    tmp30 = tmp15 - tmp24
    tmp31 = tmp30 * tmp30
    tmp32 = tmp29 + tmp31
    tmp33 = tmp21 - tmp24
    tmp34 = tmp33 * tmp33
    tmp35 = tmp32 + tmp34
    tmp36 = tmp35 / tmp23
    tl.store(out_ptr0 + (x0), tmp24, xmask)
    tl.store(out_ptr1 + (x0), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jx/cjxrdhiv2f2emt2n6iyb2yuqgnvnpsttu5b4sh3kfwos3pxlso5j.py
# Topologically Sorted Source Nodes: [x, mul, x_att], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   mul => mul
#   x => add
#   x_att => add_1, add_2, mul_1, mul_2, rsqrt, sub
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %slice_2), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %primals_3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %getitem_1), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %primals_4), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %primals_5), kwargs = {})
triton_poi_fused_add_mul_native_layer_norm_1 = async_compile.triton('triton_poi_fused_add_mul_native_layer_norm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp6 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mt/cmtozhm7q5lcnrocrirlhrqhrzplpvq4y265l5hvavi5s567upvt.py
# Topologically Sorted Source Nodes: [scores, eq, scores_1, p_attn], Original ATen: [aten.div, aten.eq, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   eq => eq
#   p_attn => amax, exp, sub_1, sum_1
#   scores => div
#   scores_1 => full_default, where
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_11, 1.0), kwargs = {})
#   %eq : [num_users=4] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze, 0), kwargs = {})
#   %full_default : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([], -1000000000.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %div), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
triton_poi_fused__softmax_div_eq_masked_fill_2 = async_compile.triton('triton_poi_fused__softmax_div_eq_masked_fill_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_div_eq_masked_fill_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_div_eq_masked_fill_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr1 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = -1000000000.0
    tmp7 = tl.where(tmp2, tmp6, tmp5)
    tmp9 = tmp8 == tmp1
    tmp11 = tmp10 * tmp4
    tmp12 = tl.where(tmp9, tmp6, tmp11)
    tmp13 = triton_helpers.maximum(tmp7, tmp12)
    tmp15 = tmp14 == tmp1
    tmp17 = tmp16 * tmp4
    tmp18 = tl.where(tmp15, tmp6, tmp17)
    tmp19 = triton_helpers.maximum(tmp13, tmp18)
    tmp21 = tmp20 == tmp1
    tmp23 = tmp22 * tmp4
    tmp24 = tl.where(tmp21, tmp6, tmp23)
    tmp25 = triton_helpers.maximum(tmp19, tmp24)
    tmp26 = tmp7 - tmp25
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tmp12 - tmp25
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp18 - tmp25
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp30 + tmp32
    tmp34 = tmp24 - tmp25
    tmp35 = tl_math.exp(tmp34)
    tmp36 = tmp33 + tmp35
    tl.store(out_ptr0 + (x2), tmp25, xmask)
    tl.store(out_ptr1 + (x2), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v3/cv3a53fndricu3fucyvy74ps57trral2usqae4ewnas6i4ionghe.py
# Topologically Sorted Source Nodes: [scores, eq, scores_1, p_attn], Original ATen: [aten.div, aten.eq, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   eq => eq
#   p_attn => amax, div_1, exp, sub_1
#   scores => div
#   scores_1 => full_default, where
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_11, 1.0), kwargs = {})
#   %eq : [num_users=4] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze, 0), kwargs = {})
#   %full_default : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([], -1000000000.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %div), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_div_eq_masked_fill_3 = async_compile.triton('triton_poi_fused__softmax_div_eq_masked_fill_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_div_eq_masked_fill_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_div_eq_masked_fill_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex // 16
    x3 = xindex
    x4 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp8 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = -1000000000.0
    tmp7 = tl.where(tmp2, tmp6, tmp5)
    tmp9 = tmp7 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp12 = tmp10 / tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2r/c2rlodbq6jhhswcq6jueytyqdtiqyted6c3rpscxymva3b55etsf.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x1), xmask & ymask)
    tl.store(out_ptr0 + (x1 + 4*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ce/ccemjzxnprgodfvpwnfi2jrcy5q6ayg6uf7iw3ea4rals3kyl3dj.py
# Topologically Sorted Source Nodes: [x, x_4, mul_1], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_3
#   x => add
#   x_4 => add_3
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %slice_2), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_17), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %primals_3), kwargs = {})
triton_poi_fused_add_mul_5 = async_compile.triton('triton_poi_fused_add_mul_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jk/cjkhg4mr2pcjyyrzm336h3fpcqnxjivgikxjssbhepxkj22nm6te.py
# Topologically Sorted Source Nodes: [x_affine], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_affine => add_4, rsqrt_1, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
triton_poi_fused_native_layer_norm_6 = async_compile.triton('triton_poi_fused_native_layer_norm_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hr/chrwfmf5n6lytleuobxdv2bgzpvakzttxmjlzy5jrs324ji5p3f3.py
# Topologically Sorted Source Nodes: [x_affine], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_affine => add_4, add_5, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, %getitem_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %primals_14), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %primals_15), kwargs = {})
triton_poi_fused_native_layer_norm_7 = async_compile.triton('triton_poi_fused_native_layer_norm_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r5/cr53zzbpaogm575igajqrvjxmywgtmm2s2cyz3jwrtbn3346ad4n.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_19,), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_8 = async_compile.triton('triton_poi_fused_relu_threshold_backward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_8(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xh/cxhr7cpf2jkl2btb2z2uw5gpj62b2o34ergpwerkwwncyr6eka3h.py
# Topologically Sorted Source Nodes: [x, x_4, x_5, mul_2], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   mul_2 => mul_6
#   x => add
#   x_4 => add_3
#   x_5 => add_6
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %slice_2), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_17), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_21), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %primals_3), kwargs = {})
triton_poi_fused_add_mul_9 = async_compile.triton('triton_poi_fused_add_mul_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2), xmask)
    tmp8 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
    tl.store(out_ptr0 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uu/cuuiof3nvl56b6vwgi4ry4rhebqa6hwcdfztkkiagq36ew6brzhd.py
# Topologically Sorted Source Nodes: [x_8, mul_3], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   mul_3 => mul_9
#   x_8 => add_9
# Graph fragment:
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_39), kwargs = {})
#   %mul_9 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %primals_3), kwargs = {})
triton_poi_fused_add_mul_10 = async_compile.triton('triton_poi_fused_add_mul_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4q/c4qavhczbojivok2tgrh7tly44kfiq62yztg3oipycd3lvlyopco.py
# Topologically Sorted Source Nodes: [x_8, x_9, mul_4], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   mul_4 => mul_12
#   x_8 => add_9
#   x_9 => add_12
# Graph fragment:
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_39), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_43), kwargs = {})
#   %mul_12 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %primals_3), kwargs = {})
triton_poi_fused_add_mul_11 = async_compile.triton('triton_poi_fused_add_mul_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jo/cjoofvnwpi4vemkhe7cregyngbs34uxkkswn6jk6iywspif2oogi.py
# Topologically Sorted Source Nodes: [x_16, x_17, mul_8], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   mul_8 => mul_24
#   x_16 => add_21
#   x_17 => add_24
# Graph fragment:
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_83), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_87), kwargs = {})
#   %mul_24 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %primals_3), kwargs = {})
triton_poi_fused_add_mul_12 = async_compile.triton('triton_poi_fused_add_mul_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69 = args
    args.clear()
    assert_size_stride(primals_1, (1, 5000, 4), (20000, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4), (4, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4), (4, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 4), (4, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4), (4, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, ), (1, ))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, 4), (4, 1))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, 4), (4, 1))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, 4), (4, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, 4), (4, 1))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, 4), (4, 1))
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (4, 4), (4, 1))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (4, 4), (4, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, 4), (4, 1))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, ), (1, ))
    assert_size_stride(primals_38, (4, 4), (4, 1))
    assert_size_stride(primals_39, (4, ), (1, ))
    assert_size_stride(primals_40, (4, 4), (4, 1))
    assert_size_stride(primals_41, (4, ), (1, ))
    assert_size_stride(primals_42, (4, 4), (4, 1))
    assert_size_stride(primals_43, (4, ), (1, ))
    assert_size_stride(primals_44, (4, 4), (4, 1))
    assert_size_stride(primals_45, (4, ), (1, ))
    assert_size_stride(primals_46, (4, ), (1, ))
    assert_size_stride(primals_47, (4, ), (1, ))
    assert_size_stride(primals_48, (4, 4), (4, 1))
    assert_size_stride(primals_49, (4, ), (1, ))
    assert_size_stride(primals_50, (4, 4), (4, 1))
    assert_size_stride(primals_51, (4, ), (1, ))
    assert_size_stride(primals_52, (4, ), (1, ))
    assert_size_stride(primals_53, (4, ), (1, ))
    assert_size_stride(primals_54, (4, 4), (4, 1))
    assert_size_stride(primals_55, (4, ), (1, ))
    assert_size_stride(primals_56, (4, 4), (4, 1))
    assert_size_stride(primals_57, (4, ), (1, ))
    assert_size_stride(primals_58, (4, 4), (4, 1))
    assert_size_stride(primals_59, (4, ), (1, ))
    assert_size_stride(primals_60, (4, 4), (4, 1))
    assert_size_stride(primals_61, (4, ), (1, ))
    assert_size_stride(primals_62, (4, ), (1, ))
    assert_size_stride(primals_63, (4, ), (1, ))
    assert_size_stride(primals_64, (4, 4), (4, 1))
    assert_size_stride(primals_65, (4, ), (1, ))
    assert_size_stride(primals_66, (4, 4), (4, 1))
    assert_size_stride(primals_67, (4, ), (1, ))
    assert_size_stride(primals_68, (4, ), (1, ))
    assert_size_stride(primals_69, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 4, 1), (4, 1, 4), torch.float32)
        buf1 = empty_strided_cuda((1, 4, 1), (4, 1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [x, mul, x_att], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_native_layer_norm_0.run(primals_2, primals_1, primals_3, buf0, buf1, 4, grid=grid(4), stream=stream0)
        buf2 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, mul, x_att], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_native_layer_norm_1.run(primals_2, primals_1, primals_3, buf0, buf1, primals_4, primals_5, buf2, buf3, 16, grid=grid(16), stream=stream0)
        del primals_4
        del primals_5
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, reinterpret_tensor(buf3, (4, 4), (4, 1), 0), reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf4)
        del primals_7
        buf5 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf3, (4, 4), (4, 1), 0), reinterpret_tensor(primals_8, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf5)
        del primals_9
        buf6 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, reinterpret_tensor(buf3, (4, 4), (4, 1), 0), reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf6)
        del primals_11
        buf7 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (4, 4, 1), (1, 4, 0), 0), reinterpret_tensor(buf5, (4, 1, 4), (1, 0, 4), 0), out=buf7)
        buf8 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        buf9 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [scores, eq, scores_1, p_attn], Original ATen: [aten.div, aten.eq, aten.masked_fill, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_eq_masked_fill_2.run(primals_3, buf7, buf8, buf9, 16, grid=grid(16), stream=stream0)
        buf10 = reinterpret_tensor(buf7, (1, 4, 4, 4), (64, 16, 4, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [scores, eq, scores_1, p_attn], Original ATen: [aten.div, aten.eq, aten.masked_fill, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_eq_masked_fill_3.run(buf10, primals_3, buf8, buf9, 64, grid=grid(64), stream=stream0)
        buf11 = reinterpret_tensor(buf9, (4, 4, 1), (4, 1, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf6, (4, 4, 1), (1, 4, 0), 0), out=buf11)
        buf12 = reinterpret_tensor(buf8, (1, 4, 4, 1), (16, 4, 1, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf11, buf12, 4, 4, grid=grid(4, 4), stream=stream0)
        buf13 = reinterpret_tensor(buf11, (4, 4), (4, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_att_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf12, (4, 4), (4, 1), 0), reinterpret_tensor(primals_12, (4, 4), (1, 4), 0), out=buf13)
        buf14 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_4, mul_1], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_5.run(primals_2, primals_1, buf13, primals_13, primals_3, buf14, 16, grid=grid(16), stream=stream0)
        buf15 = buf1; del buf1  # reuse
        buf16 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_affine], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_6.run(buf14, buf15, buf16, 4, grid=grid(4), stream=stream0)
        buf17 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_affine], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7.run(buf14, buf15, buf16, primals_14, primals_15, buf17, 16, grid=grid(16), stream=stream0)
        del primals_15
        buf18 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf17, (4, 4), (4, 1), 0), reinterpret_tensor(primals_16, (4, 4), (1, 4), 0), out=buf18)
        buf19 = reinterpret_tensor(buf18, (1, 4, 4), (16, 4, 1), 0); del buf18  # reuse
        buf94 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_8.run(buf19, primals_17, buf94, 16, grid=grid(16), stream=stream0)
        del primals_17
        buf20 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_affine_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf19, (4, 4), (4, 1), 0), reinterpret_tensor(primals_18, (4, 4), (1, 4), 0), out=buf20)
        buf21 = reinterpret_tensor(buf13, (1, 4, 4), (16, 4, 1), 0); del buf13  # reuse
        buf22 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_4, x_5, mul_2], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_9.run(buf21, primals_2, primals_1, primals_13, buf20, primals_19, primals_3, buf22, 16, grid=grid(16), stream=stream0)
        del primals_1
        del primals_13
        del primals_19
        del primals_2
        buf23 = buf16; del buf16  # reuse
        buf24 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_att_2], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_6.run(buf22, buf23, buf24, 4, grid=grid(4), stream=stream0)
        buf25 = reinterpret_tensor(buf20, (1, 4, 4), (16, 4, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_att_2], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7.run(buf22, buf23, buf24, primals_20, primals_21, buf25, 16, grid=grid(16), stream=stream0)
        del primals_21
        buf26 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_23, reinterpret_tensor(buf25, (4, 4), (4, 1), 0), reinterpret_tensor(primals_22, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf26)
        del primals_23
        buf27 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_25, reinterpret_tensor(buf25, (4, 4), (4, 1), 0), reinterpret_tensor(primals_24, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf27)
        del primals_25
        buf28 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_27, reinterpret_tensor(buf25, (4, 4), (4, 1), 0), reinterpret_tensor(primals_26, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf28)
        del primals_27
        buf29 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (4, 4, 1), (1, 4, 0), 0), reinterpret_tensor(buf27, (4, 1, 4), (1, 0, 4), 0), out=buf29)
        buf30 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        buf31 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [eq, scores_1, scores_2, scores_3, p_attn_2], Original ATen: [aten.eq, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_eq_masked_fill_2.run(primals_3, buf29, buf30, buf31, 16, grid=grid(16), stream=stream0)
        buf32 = reinterpret_tensor(buf29, (1, 4, 4, 4), (64, 16, 4, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [eq, scores_1, scores_2, scores_3, p_attn_2], Original ATen: [aten.eq, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_eq_masked_fill_3.run(buf32, primals_3, buf30, buf31, 64, grid=grid(64), stream=stream0)
        buf33 = reinterpret_tensor(buf31, (4, 4, 1), (4, 1, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf28, (4, 4, 1), (1, 4, 0), 0), out=buf33)
        buf34 = reinterpret_tensor(buf30, (1, 4, 4, 1), (16, 4, 1, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf33, buf34, 4, 4, grid=grid(4, 4), stream=stream0)
        buf35 = reinterpret_tensor(buf33, (4, 4), (4, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_att_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf34, (4, 4), (4, 1), 0), reinterpret_tensor(primals_28, (4, 4), (1, 4), 0), out=buf35)
        buf36 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8, mul_3], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_10.run(buf21, buf35, primals_29, primals_3, buf36, 16, grid=grid(16), stream=stream0)
        buf37 = buf24; del buf24  # reuse
        buf38 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_affine_2], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_6.run(buf36, buf37, buf38, 4, grid=grid(4), stream=stream0)
        buf39 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_affine_2], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7.run(buf36, buf37, buf38, primals_30, primals_31, buf39, 16, grid=grid(16), stream=stream0)
        del primals_31
        buf40 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf39, (4, 4), (4, 1), 0), reinterpret_tensor(primals_32, (4, 4), (1, 4), 0), out=buf40)
        buf41 = reinterpret_tensor(buf40, (1, 4, 4), (16, 4, 1), 0); del buf40  # reuse
        buf93 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_1], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_8.run(buf41, primals_33, buf93, 16, grid=grid(16), stream=stream0)
        del primals_33
        buf42 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_affine_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf41, (4, 4), (4, 1), 0), reinterpret_tensor(primals_34, (4, 4), (1, 4), 0), out=buf42)
        buf43 = buf21; del buf21  # reuse
        buf44 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8, x_9, mul_4], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_11.run(buf43, buf35, primals_29, buf42, primals_35, primals_3, buf44, 16, grid=grid(16), stream=stream0)
        del primals_29
        del primals_35
        buf45 = buf38; del buf38  # reuse
        buf46 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_att_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_6.run(buf44, buf45, buf46, 4, grid=grid(4), stream=stream0)
        buf47 = reinterpret_tensor(buf42, (1, 4, 4), (16, 4, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_att_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7.run(buf44, buf45, buf46, primals_36, primals_37, buf47, 16, grid=grid(16), stream=stream0)
        del primals_37
        buf48 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_39, reinterpret_tensor(buf47, (4, 4), (4, 1), 0), reinterpret_tensor(primals_38, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf48)
        del primals_39
        buf49 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_41, reinterpret_tensor(buf47, (4, 4), (4, 1), 0), reinterpret_tensor(primals_40, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf49)
        del primals_41
        buf50 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_43, reinterpret_tensor(buf47, (4, 4), (4, 1), 0), reinterpret_tensor(primals_42, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf50)
        del primals_43
        buf51 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (4, 4, 1), (1, 4, 0), 0), reinterpret_tensor(buf49, (4, 1, 4), (1, 0, 4), 0), out=buf51)
        buf52 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        buf53 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [eq, scores_1, scores_4, scores_5, p_attn_4], Original ATen: [aten.eq, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_eq_masked_fill_2.run(primals_3, buf51, buf52, buf53, 16, grid=grid(16), stream=stream0)
        buf54 = reinterpret_tensor(buf51, (1, 4, 4, 4), (64, 16, 4, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [eq, scores_1, scores_4, scores_5, p_attn_4], Original ATen: [aten.eq, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_eq_masked_fill_3.run(buf54, primals_3, buf52, buf53, 64, grid=grid(64), stream=stream0)
        buf55 = reinterpret_tensor(buf53, (4, 4, 1), (4, 1, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf50, (4, 4, 1), (1, 4, 0), 0), out=buf55)
        buf56 = reinterpret_tensor(buf52, (1, 4, 4, 1), (16, 4, 1, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf55, buf56, 4, 4, grid=grid(4, 4), stream=stream0)
        buf57 = reinterpret_tensor(buf55, (4, 4), (4, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_att_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf56, (4, 4), (4, 1), 0), reinterpret_tensor(primals_44, (4, 4), (1, 4), 0), out=buf57)
        buf58 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, mul_5], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_10.run(buf43, buf57, primals_45, primals_3, buf58, 16, grid=grid(16), stream=stream0)
        buf59 = buf46; del buf46  # reuse
        buf60 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_affine_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_6.run(buf58, buf59, buf60, 4, grid=grid(4), stream=stream0)
        buf61 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_affine_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7.run(buf58, buf59, buf60, primals_46, primals_47, buf61, 16, grid=grid(16), stream=stream0)
        del primals_47
        buf62 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf61, (4, 4), (4, 1), 0), reinterpret_tensor(primals_48, (4, 4), (1, 4), 0), out=buf62)
        buf63 = reinterpret_tensor(buf62, (1, 4, 4), (16, 4, 1), 0); del buf62  # reuse
        buf92 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_8.run(buf63, primals_49, buf92, 16, grid=grid(16), stream=stream0)
        del primals_49
        buf64 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_affine_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf63, (4, 4), (4, 1), 0), reinterpret_tensor(primals_50, (4, 4), (1, 4), 0), out=buf64)
        buf65 = buf43; del buf43  # reuse
        buf66 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13, mul_6], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_11.run(buf65, buf57, primals_45, buf64, primals_51, primals_3, buf66, 16, grid=grid(16), stream=stream0)
        del primals_45
        del primals_51
        buf67 = buf60; del buf60  # reuse
        buf68 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_att_6], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_6.run(buf66, buf67, buf68, 4, grid=grid(4), stream=stream0)
        buf69 = reinterpret_tensor(buf64, (1, 4, 4), (16, 4, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_att_6], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7.run(buf66, buf67, buf68, primals_52, primals_53, buf69, 16, grid=grid(16), stream=stream0)
        del primals_53
        buf70 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_55, reinterpret_tensor(buf69, (4, 4), (4, 1), 0), reinterpret_tensor(primals_54, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf70)
        del primals_55
        buf71 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_57, reinterpret_tensor(buf69, (4, 4), (4, 1), 0), reinterpret_tensor(primals_56, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf71)
        del primals_57
        buf72 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_59, reinterpret_tensor(buf69, (4, 4), (4, 1), 0), reinterpret_tensor(primals_58, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf72)
        del primals_59
        buf73 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (4, 4, 1), (1, 4, 0), 0), reinterpret_tensor(buf71, (4, 1, 4), (1, 0, 4), 0), out=buf73)
        buf74 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        buf75 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [eq, scores_1, scores_6, scores_7, p_attn_6], Original ATen: [aten.eq, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_eq_masked_fill_2.run(primals_3, buf73, buf74, buf75, 16, grid=grid(16), stream=stream0)
        buf76 = reinterpret_tensor(buf73, (1, 4, 4, 4), (64, 16, 4, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [eq, scores_1, scores_6, scores_7, p_attn_6], Original ATen: [aten.eq, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_eq_masked_fill_3.run(buf76, primals_3, buf74, buf75, 64, grid=grid(64), stream=stream0)
        buf77 = reinterpret_tensor(buf75, (4, 4, 1), (4, 1, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf76, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf72, (4, 4, 1), (1, 4, 0), 0), out=buf77)
        buf78 = reinterpret_tensor(buf74, (1, 4, 4, 1), (16, 4, 1, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf77, buf78, 4, 4, grid=grid(4, 4), stream=stream0)
        buf79 = reinterpret_tensor(buf77, (4, 4), (4, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_att_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf78, (4, 4), (4, 1), 0), reinterpret_tensor(primals_60, (4, 4), (1, 4), 0), out=buf79)
        buf80 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_16, mul_7], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_10.run(buf65, buf79, primals_61, primals_3, buf80, 16, grid=grid(16), stream=stream0)
        buf81 = buf68; del buf68  # reuse
        buf82 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_affine_6], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_6.run(buf80, buf81, buf82, 4, grid=grid(4), stream=stream0)
        buf83 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_affine_6], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7.run(buf80, buf81, buf82, primals_62, primals_63, buf83, 16, grid=grid(16), stream=stream0)
        del primals_63
        buf84 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf83, (4, 4), (4, 1), 0), reinterpret_tensor(primals_64, (4, 4), (1, 4), 0), out=buf84)
        buf85 = reinterpret_tensor(buf84, (1, 4, 4), (16, 4, 1), 0); del buf84  # reuse
        buf91 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_3], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_8.run(buf85, primals_65, buf91, 16, grid=grid(16), stream=stream0)
        del primals_65
        buf86 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_affine_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf85, (4, 4), (4, 1), 0), reinterpret_tensor(primals_66, (4, 4), (1, 4), 0), out=buf86)
        buf87 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_16, x_17, mul_8], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_12.run(buf87, buf79, primals_61, buf86, primals_67, primals_3, 16, grid=grid(16), stream=stream0)
        del buf79
        del primals_61
        del primals_67
        buf88 = buf82; del buf82  # reuse
        buf89 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_8], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_6.run(buf87, buf88, buf89, 4, grid=grid(4), stream=stream0)
        buf90 = reinterpret_tensor(buf86, (1, 4, 4), (16, 4, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_8], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7.run(buf87, buf88, buf89, primals_68, primals_69, buf90, 16, grid=grid(16), stream=stream0)
        del buf88
        del buf89
        del primals_69
    return (buf90, primals_3, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, buf2, reinterpret_tensor(buf3, (4, 4), (4, 1), 0), buf10, reinterpret_tensor(buf12, (4, 4), (4, 1), 0), buf14, reinterpret_tensor(buf17, (4, 4), (4, 1), 0), reinterpret_tensor(buf19, (4, 4), (4, 1), 0), buf22, reinterpret_tensor(buf25, (4, 4), (4, 1), 0), buf32, reinterpret_tensor(buf34, (4, 4), (4, 1), 0), buf36, reinterpret_tensor(buf39, (4, 4), (4, 1), 0), reinterpret_tensor(buf41, (4, 4), (4, 1), 0), buf44, reinterpret_tensor(buf47, (4, 4), (4, 1), 0), buf54, reinterpret_tensor(buf56, (4, 4), (4, 1), 0), buf58, reinterpret_tensor(buf61, (4, 4), (4, 1), 0), reinterpret_tensor(buf63, (4, 4), (4, 1), 0), buf66, reinterpret_tensor(buf69, (4, 4), (4, 1), 0), buf76, reinterpret_tensor(buf78, (4, 4), (4, 1), 0), buf80, reinterpret_tensor(buf83, (4, 4), (4, 1), 0), reinterpret_tensor(buf85, (4, 4), (4, 1), 0), buf87, primals_66, buf91, primals_64, primals_60, reinterpret_tensor(buf72, (4, 1, 4), (1, 1, 4), 0), reinterpret_tensor(buf70, (4, 1, 4), (1, 1, 4), 0), reinterpret_tensor(buf71, (4, 4, 1), (1, 4, 1), 0), primals_58, primals_56, primals_54, primals_50, buf92, primals_48, primals_44, reinterpret_tensor(buf50, (4, 1, 4), (1, 1, 4), 0), reinterpret_tensor(buf48, (4, 1, 4), (1, 1, 4), 0), reinterpret_tensor(buf49, (4, 4, 1), (1, 4, 1), 0), primals_42, primals_40, primals_38, primals_34, buf93, primals_32, primals_28, reinterpret_tensor(buf28, (4, 1, 4), (1, 1, 4), 0), reinterpret_tensor(buf26, (4, 1, 4), (1, 1, 4), 0), reinterpret_tensor(buf27, (4, 4, 1), (1, 4, 1), 0), primals_26, primals_24, primals_22, primals_18, buf94, primals_16, primals_12, reinterpret_tensor(buf6, (4, 1, 4), (1, 1, 4), 0), reinterpret_tensor(buf4, (4, 1, 4), (1, 1, 4), 0), reinterpret_tensor(buf5, (4, 4, 1), (1, 4, 1), 0), primals_10, primals_8, primals_6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 5000, 4), (20000, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
