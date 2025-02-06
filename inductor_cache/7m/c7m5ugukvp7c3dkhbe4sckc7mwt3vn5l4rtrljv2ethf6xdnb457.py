# AOT ID: ['12_inference']
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


# kernel path: inductor_cache/2m/c2mlnkdeaa45hk7jb7u2hbnduvhsxpnf4x3adh3pdzqqycrog5ua.py
# Topologically Sorted Source Nodes: [mask, eq_1, setitem], Original ATen: [aten.eq, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   eq_1 => eq_1
#   mask => eq
#   setitem => full_default, index_put
# Graph fragment:
#   %eq : [num_users=2] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg1_1, 255), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%eq, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : [num_users=4] = call_function[target=torch.ops.aten.index_put.default](args = (%arg1_1, [%eq], %full_default), kwargs = {})
triton_poi_fused_eq_index_put_lift_fresh_0 = async_compile.triton('triton_poi_fused_eq_index_put_lift_fresh_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i1', 'out_ptr1': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_index_put_lift_fresh_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eq_index_put_lift_fresh_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 255, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 == tmp4
    tmp6 = tl.where(tmp2, tmp4, tmp0)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qg/cqgybhcejvn5ygkdk7vp5h4i5lm32acthwagjjvxytagt6qfqpex.py
# Topologically Sorted Source Nodes: [logits, setitem_1, cross_entropy], Original ATen: [aten._to_copy, aten.lift_fresh, aten.index_put, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy => amax_2, sub_4
#   logits => convert_element_type
#   setitem_1 => full_default_2, index_put_1
# Graph fragment:
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1000000.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_1 : [num_users=4] = call_function[target=torch.ops.aten.index_put.default](args = (%convert_element_type, [%convert_element_type_1], %full_default_2), kwargs = {})
#   %amax_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type, [1], True), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %amax_2), kwargs = {})
triton_poi_fused__log_softmax__to_copy_index_put_lift_fresh_1 = async_compile.triton('triton_poi_fused__log_softmax__to_copy_index_put_lift_fresh_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__to_copy_index_put_lift_fresh_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__to_copy_index_put_lift_fresh_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x2), xmask)
    tmp11 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (tmp5 != 0)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = -1000000.0
    tmp10 = tl.where(tmp6, tmp9, tmp8)
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = triton_helpers.maximum(tmp12, tmp14)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = triton_helpers.maximum(tmp15, tmp17)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = triton_helpers.maximum(tmp18, tmp20)
    tmp22 = tmp8 - tmp21
    tl.store(out_ptr0 + (x2), tmp10, xmask)
    tl.store(out_ptr1 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xw/cxwimi5dfjkuizhhxupw2nspvkqauxsbhmrs35etzvkgxjptpbmd.py
# Topologically Sorted Source Nodes: [q, log_q], Original ATen: [aten._softmax, aten._log_softmax]
# Source node to ATen node mapping:
#   log_q => amax_1, sub_1
#   q => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%index_put_1, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index_put_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%index_put_1, [1], True), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index_put_1, %amax_1), kwargs = {})
triton_poi_fused__log_softmax__softmax_2 = async_compile.triton('triton_poi_fused__log_softmax__softmax_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__softmax_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__softmax_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
    tl.store(out_ptr1 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uz/cuz37itdn43h4za53elasixosjvd5g3zglm3y6gfoo57gmxcfeb2.py
# Topologically Sorted Source Nodes: [q, sub_, log_q, mul_, losses, setitem_2], Original ATen: [aten._softmax, aten.sub, aten._log_softmax, aten.mul, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   log_q => exp_1, log, sub_2, sum_2
#   losses => mul_1
#   mul_ => mul
#   q => div, sum_1
#   setitem_2 => full_default_3, index_put_2
#   sub_ => sub_3
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, 0.3333333333333333), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %log), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %sub_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 0.15), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_2 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%mul_1, [%convert_element_type_2], %full_default_3), kwargs = {})
triton_poi_fused__log_softmax__softmax_index_put_lift_fresh_mul_sub_3 = async_compile.triton('triton_poi_fused__log_softmax__softmax_index_put_lift_fresh_mul_sub_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__softmax_index_put_lift_fresh_mul_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__softmax_index_put_lift_fresh_mul_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tmp9 = 0.3333333333333333
    tmp10 = tmp8 - tmp9
    tmp13 = tl_math.exp(tmp12)
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp13 + tmp15
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp16 + tmp18
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp19 + tmp21
    tmp23 = tl_math.log(tmp22)
    tmp24 = tmp11 - tmp23
    tmp25 = tmp10 * tmp24
    tmp27 = x0
    tmp28 = tmp26 == tmp27
    tmp29 = 1.0
    tmp30 = 0.0
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = (tmp31 != 0)
    tmp33 = 0.15
    tmp34 = tmp25 * tmp33
    tmp35 = tl.where(tmp32, tmp30, tmp34)
    tl.store(in_out_ptr0 + (x2), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6q/c6qszd2mge33jf64hj7umy5sewju2cmy5xu4ij5zc5yjxg3iolgm.py
# Topologically Sorted Source Nodes: [sum_1, cross_entropy, losses_1], Original ATen: [aten.sum, aten.nll_loss_forward, aten.add]
# Source node to ATen node mapping:
#   cross_entropy => full_default_5, ne_1, neg, where_1
#   losses_1 => add
#   sum_1 => sum_3
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%index_put_2, [1]), kwargs = {})
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%index_put, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_5), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, %where_1), kwargs = {})
triton_poi_fused_add_nll_loss_forward_sum_4 = async_compile.triton('triton_poi_fused_add_nll_loss_forward_sum_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_nll_loss_forward_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_nll_loss_forward_sum_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x0), xmask)
    tmp18 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tl.full([1], -100, tl.int64)
    tmp9 = tmp7 != tmp8
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tl.where(tmp9, tmp7, tmp10)
    tmp12 = tl.full([XBLOCK], 4, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tl.device_assert(((0 <= tmp15) & (tmp15 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp15 < 4")
    tmp17 = tl.load(in_ptr2 + (tmp15 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl_math.exp(tmp18)
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp19 + tmp21
    tmp24 = tl_math.exp(tmp23)
    tmp25 = tmp22 + tmp24
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tmp25 + tmp27
    tmp29 = tl_math.log(tmp28)
    tmp30 = tmp17 - tmp29
    tmp31 = -tmp30
    tmp32 = 0.0
    tmp33 = tl.where(tmp9, tmp31, tmp32)
    tmp34 = tmp6 + tmp33
    tl.store(out_ptr0 + (x0), tmp34, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.bool)
        buf1 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [mask, eq_1, setitem], Original ATen: [aten.eq, aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_eq_index_put_lift_fresh_0.run(arg1_1, buf0, buf1, 4, grid=grid(4), stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits, setitem_1, cross_entropy], Original ATen: [aten._to_copy, aten.lift_fresh, aten.index_put, aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__to_copy_index_put_lift_fresh_1.run(buf1, arg0_1, buf2, buf7, 16, grid=grid(16), stream=stream0)
        del arg0_1
        buf3 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q, log_q], Original ATen: [aten._softmax, aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__softmax_2.run(buf2, buf3, buf4, 16, grid=grid(16), stream=stream0)
        buf5 = buf2; del buf2  # reuse
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [q, sub_, log_q, mul_, losses, setitem_2], Original ATen: [aten._softmax, aten.sub, aten._log_softmax, aten.mul, aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__softmax_index_put_lift_fresh_mul_sub_3.run(buf6, buf3, buf4, buf1, 16, grid=grid(16), stream=stream0)
        del buf3
        del buf4
        buf8 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sum_1, cross_entropy, losses_1], Original ATen: [aten.sum, aten.nll_loss_forward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_nll_loss_forward_sum_4.run(buf6, buf1, buf7, buf8, 4, grid=grid(4), stream=stream0)
        del buf1
        del buf6
        del buf7
    return (buf0, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
