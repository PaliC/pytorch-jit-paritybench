# AOT ID: ['4_inference']
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


# kernel path: inductor_cache/e3/ce3izi3uq2fi4p6p4uhrijawxmq7yiw7nl6ukyi62r3j4somdgii.py
# Topologically Sorted Source Nodes: [mul, mul_1, add, mul_2, mul_3, sub, mul_4, mul_5, add_1, mul_6, mul_7, sub_1, mul_8, mul_9, add_2, mul_10, mul_11, sub_2, mul_12, mul_13, add_3, mul_14, mul_15, sub_3], Original ATen: [aten.mul, aten.add, aten.sub]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %select_4), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %select_6), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %select_8), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, %select_10), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_21, %select_22), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_23, %select_24), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_25, %select_26), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_27, %select_28), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_40, %select_41), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_42, %select_43), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_44, %select_45), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_46, %select_47), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_59, %select_60), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_61, %select_62), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_63, %select_64), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_65, %select_66), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_14, %mul_15), kwargs = {})
triton_poi_fused_add_mul_sub_0 = async_compile.triton('triton_poi_fused_add_mul_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sub_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x2 = xindex // 12
    x3 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (2*x0 + 24*x2), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp0 >= tmp2
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp0 < tmp6
    tmp8 = tl.load(in_ptr1 + (1 + 2*x0 + 24*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.where(tmp3, tmp4, tmp8)
    tmp10 = tl.load(in_ptr0 + (2*x3), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + (1 + 2*x3), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.where(tmp3, tmp10, tmp11)
    tmp13 = tmp9 * tmp12
    tmp14 = tmp2 >= tmp0
    tmp15 = tmp2 < tmp2
    tmp16 = tl.load(in_ptr0 + (2*x0 + 24*x2), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp2 >= tmp2
    tmp18 = tmp2 < tmp6
    tmp19 = tl.load(in_ptr1 + (1 + 2*x0 + 24*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp15, tmp16, tmp19)
    tmp21 = tl.load(in_ptr0 + (2*x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr1 + (1 + 2*x3), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp15, tmp21, tmp22)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp13 + tmp24
    tmp26 = tmp20 * tmp12
    tmp27 = tmp9 * tmp23
    tmp28 = tmp26 - tmp27
    tmp29 = tl.load(in_ptr0 + (6 + 2*x0 + 24*x2), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr1 + (7 + 2*x0 + 24*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.where(tmp3, tmp29, tmp30)
    tmp32 = tmp31 * tmp12
    tmp33 = tl.load(in_ptr0 + (6 + 2*x0 + 24*x2), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr1 + (7 + 2*x0 + 24*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tmp35 * tmp23
    tmp37 = tmp32 + tmp36
    tmp38 = tmp35 * tmp12
    tmp39 = tmp31 * tmp23
    tmp40 = tmp38 - tmp39
    tmp41 = tl.load(in_ptr0 + (12 + 2*x0 + 24*x2), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr1 + (13 + 2*x0 + 24*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.where(tmp3, tmp41, tmp42)
    tmp44 = tmp43 * tmp12
    tmp45 = tl.load(in_ptr0 + (12 + 2*x0 + 24*x2), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr1 + (13 + 2*x0 + 24*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.where(tmp15, tmp45, tmp46)
    tmp48 = tmp47 * tmp23
    tmp49 = tmp44 + tmp48
    tmp50 = tmp47 * tmp12
    tmp51 = tmp43 * tmp23
    tmp52 = tmp50 - tmp51
    tmp53 = tl.load(in_ptr0 + (18 + 2*x0 + 24*x2), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr1 + (19 + 2*x0 + 24*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tl.where(tmp3, tmp53, tmp54)
    tmp56 = tmp55 * tmp12
    tmp57 = tl.load(in_ptr0 + (18 + 2*x0 + 24*x2), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr1 + (19 + 2*x0 + 24*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tl.where(tmp15, tmp57, tmp58)
    tmp60 = tmp59 * tmp23
    tmp61 = tmp56 + tmp60
    tmp62 = tmp59 * tmp12
    tmp63 = tmp55 * tmp23
    tmp64 = tmp62 - tmp63
    tl.store(out_ptr0 + (x3), tmp25, xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
    tl.store(out_ptr2 + (x3), tmp37, xmask)
    tl.store(out_ptr3 + (x3), tmp40, xmask)
    tl.store(out_ptr4 + (x3), tmp49, xmask)
    tl.store(out_ptr5 + (x3), tmp52, xmask)
    tl.store(out_ptr6 + (x3), tmp61, xmask)
    tl.store(out_ptr7 + (x3), tmp64, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i7/ci7pk7ypttzzox3gcrft3duhvralwzeh7vb7wmaaytseoibo43ne.py
# Topologically Sorted Source Nodes: [gcc_fft_batch], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   gcc_fft_batch => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_3, %unsqueeze_4], -1), kwargs = {})
triton_poi_fused_stack_1 = async_compile.triton('triton_poi_fused_stack_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vz/cvzjm2gmf5ssysvvk47ywixxoytqhp5d4mwcr7hxe2jif2krpla2.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   setitem_3 => copy_3
# Graph fragment:
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_31, %slice_27), kwargs = {})
#   %slice_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_3, %copy_3, 3, -2, 9223372036854775807), kwargs = {})
triton_poi_fused_copy_2 = async_compile.triton('triton_poi_fused_copy_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x3 = xindex // 5
    x2 = xindex // 20
    x4 = (xindex % 20)
    x5 = xindex
    tmp13 = tl.load(in_ptr2 + (x4 + 80*x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-1) + x0 + 4*x3), tmp2 & xmask, other=0.0)
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp4 == tmp4
    tmp6 = tmp0 < tmp1
    tmp7 = tl.load(in_ptr0 + (x0 + 4*x3), tmp6 & xmask, other=0.0)
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = tmp4 == tmp8
    tmp10 = tl.load(in_ptr1 + ((-1) + x0 + 4*x3), tmp2 & xmask, other=0.0)
    tmp11 = tmp8 == tmp8
    tmp12 = tl.load(in_ptr1 + (x0 + 4*x3), tmp6 & xmask, other=0.0)
    tmp14 = tl.where(tmp6, tmp12, tmp13)
    tmp15 = float("nan")
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp2, tmp10, tmp16)
    tmp18 = tl.where(tmp9, tmp14, tmp15)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tl.where(tmp6, tmp7, tmp19)
    tmp21 = tl.where(tmp5, tmp20, tmp19)
    tmp22 = tl.where(tmp2, tmp3, tmp21)
    tl.store(out_ptr0 + (x5), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dw/cdw3b6yyyugc3z5l6t6qruups3fhihz2cirunvuqeaxs6p7dyjub.py
# Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   setitem => copy
#   setitem_1 => copy_1
#   setitem_2 => copy_2
#   setitem_3 => copy_3
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %slice_3), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int, %copy, 3, 0, 3), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%empty, %slice_scatter_default, 2, 0), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_13, %slice_9), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_1, %copy_1, 3, -2, 9223372036854775807), kwargs = {})
#   %select_scatter_default_1 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %slice_scatter_default_1, 2, 0), kwargs = {})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_23, %slice_19), kwargs = {})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_2, %copy_2, 3, 0, 3), kwargs = {})
#   %select_scatter_default_2 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %slice_scatter_default_2, 2, 1), kwargs = {})
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_31, %slice_27), kwargs = {})
#   %slice_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_3, %copy_3, 3, -2, 9223372036854775807), kwargs = {})
#   %select_scatter_default_3 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_2, %slice_scatter_default_3, 2, 1), kwargs = {})
triton_poi_fused_copy_3 = async_compile.triton('triton_poi_fused_copy_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 20) % 4)
    x3 = xindex // 80
    x4 = (xindex % 20)
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 4)
    x5 = xindex
    tmp3 = tl.load(in_ptr0 + (x4 + 20*x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x4 + 80*x3), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = x0
    tmp5 = tl.full([1], 3, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr1 + (x0 + 4*x1 + 16*x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = tmp1 == tmp8
    tmp10 = tmp4 >= tmp5
    tmp11 = tl.load(in_ptr2 + ((-1) + x0 + 4*x1 + 16*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp8 == tmp8
    tmp13 = tl.load(in_ptr2 + (x0 + 4*x1 + 16*x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = float("nan")
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.where(tmp10, tmp11, tmp17)
    tmp19 = tl.where(tmp9, tmp15, tmp16)
    tmp20 = tl.where(tmp9, tmp18, tmp19)
    tmp21 = tl.where(tmp6, tmp7, tmp20)
    tmp22 = tmp0 == tmp8
    tmp23 = tl.where(tmp22, tmp15, tmp16)
    tmp24 = tl.where(tmp22, tmp18, tmp23)
    tmp25 = tl.where(tmp2, tmp21, tmp24)
    tmp26 = tl.where(tmp2, tmp3, tmp25)
    tl.store(out_ptr0 + (x5), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tv/ctvjns5b43qfg4om77cxnwdtsayhq75frbhle5gfbypofguoh6wt.py
# Topologically Sorted Source Nodes: [setitem_6], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   setitem_6 => copy_6
# Graph fragment:
#   %copy_6 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_59, %slice_55), kwargs = {})
#   %slice_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_6, %copy_6, 3, 0, 3), kwargs = {})
triton_poi_fused_copy_4 = async_compile.triton('triton_poi_fused_copy_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x3 = xindex // 5
    x2 = xindex // 20
    x4 = (xindex % 20)
    x5 = xindex
    tmp11 = tl.load(in_ptr2 + (40 + x4 + 80*x2), xmask)
    tmp15 = tl.load(in_ptr2 + (60 + x4 + 80*x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 4*x3), tmp2 & xmask, other=0.0)
    tmp4 = tl.full([1], 3, tl.int32)
    tmp5 = tl.full([1], 2, tl.int32)
    tmp6 = tmp4 == tmp5
    tmp7 = tmp0 >= tmp1
    tmp8 = tl.load(in_ptr1 + ((-1) + x0 + 4*x3), tmp7 & xmask, other=0.0)
    tmp9 = tmp5 == tmp5
    tmp10 = tl.load(in_ptr1 + (x0 + 4*x3), tmp2 & xmask, other=0.0)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp9, tmp12, tmp11)
    tmp14 = tl.where(tmp7, tmp8, tmp13)
    tmp16 = tl.where(tmp6, tmp12, tmp15)
    tmp17 = tl.where(tmp6, tmp14, tmp16)
    tmp18 = tl.where(tmp2, tmp3, tmp17)
    tl.store(out_ptr0 + (x5), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tr/ctral4i2vsln53fsz7txux6t7ctu5fb7aqqq6rcpj3jcunou43pm.py
# Topologically Sorted Source Nodes: [setitem_4, setitem_5, setitem_6], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   setitem_4 => copy_4
#   setitem_5 => copy_5
#   setitem_6 => copy_6
# Graph fragment:
#   %copy_4 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_41, %slice_37), kwargs = {})
#   %slice_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_4, %copy_4, 3, 0, 3), kwargs = {})
#   %select_scatter_default_4 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_3, %slice_scatter_default_4, 2, 2), kwargs = {})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_49, %slice_45), kwargs = {})
#   %slice_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_5, %copy_5, 3, -2, 9223372036854775807), kwargs = {})
#   %select_scatter_default_5 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_4, %slice_scatter_default_5, 2, 2), kwargs = {})
#   %copy_6 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_59, %slice_55), kwargs = {})
#   %slice_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_6, %copy_6, 3, 0, 3), kwargs = {})
#   %select_scatter_default_6 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_5, %slice_scatter_default_6, 2, 3), kwargs = {})
triton_poi_fused_copy_5 = async_compile.triton('triton_poi_fused_copy_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 20) % 4)
    x3 = xindex // 80
    x4 = (xindex % 20)
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 4)
    x5 = xindex
    tmp3 = tl.load(in_ptr0 + (x4 + 20*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (40 + x4 + 80*x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (x5), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 2, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 3, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.load(in_ptr1 + ((-1) + x0 + 4*x1 + 16*x3), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp4 == tmp4
    tmp11 = tmp6 < tmp7
    tmp12 = tl.load(in_ptr1 + (x0 + 4*x1 + 16*x3), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.where(tmp10, tmp14, tmp13)
    tmp16 = tl.where(tmp8, tmp9, tmp15)
    tmp18 = tl.where(tmp5, tmp14, tmp17)
    tmp19 = tl.where(tmp5, tmp16, tmp18)
    tmp20 = tl.where(tmp2, tmp3, tmp19)
    tl.store(out_ptr0 + (x5), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ba/cbascvyznixls7tne6sxr4ee2sipz4tcgiotdhyczybsjwwjmszb.py
# Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   setitem_7 => copy_7
# Graph fragment:
#   %copy_7 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_67, %slice_63), kwargs = {})
#   %slice_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_7, %copy_7, 3, -2, 9223372036854775807), kwargs = {})
#   %select_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_6, %slice_scatter_default_7, 2, 3), kwargs = {})
triton_poi_fused_copy_6 = async_compile.triton('triton_poi_fused_copy_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 20) % 4)
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 4)
    x3 = xindex // 80
    x4 = (xindex % 20)
    x5 = xindex
    tmp7 = tl.load(in_ptr1 + (60 + x4 + 80*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x5), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 3, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr0 + ((-1) + x0 + 4*x1 + 16*x3), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp10 = tl.where(tmp2, tmp8, tmp9)
    tl.store(out_ptr0 + (x5), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_fft_c], Original ATen: [aten._fft_r2c]
        buf1 = torch.ops.aten._fft_r2c.default(arg0_1, [3], 0, True)
        del arg0_1
        buf2 = buf1
        del buf1
        # Topologically Sorted Source Nodes: [getattr_1], Original ATen: [aten.view_as_real]
        buf3 = torch.ops.aten.view_as_real.default(buf2)
        buf4 = buf3
        # Topologically Sorted Source Nodes: [getattr_2], Original ATen: [aten.view_as_real]
        buf5 = torch.ops.aten.view_as_real.default(buf2)
        buf6 = buf5
        buf0 = empty_strided_cuda((4, 4, 4, 4, 5), (320, 80, 20, 5, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 4, 4, 3), (48, 12, 3, 1), torch.float32)
        buf8 = empty_strided_cuda((4, 4, 4, 3), (48, 12, 3, 1), torch.float32)
        buf14 = empty_strided_cuda((4, 4, 4, 3), (48, 12, 3, 1), torch.float32)
        buf15 = empty_strided_cuda((4, 4, 4, 3), (48, 12, 3, 1), torch.float32)
        buf23 = empty_strided_cuda((4, 4, 4, 3), (48, 12, 3, 1), torch.float32)
        buf24 = empty_strided_cuda((4, 4, 4, 3), (48, 12, 3, 1), torch.float32)
        buf30 = empty_strided_cuda((4, 4, 4, 3), (48, 12, 3, 1), torch.float32)
        buf31 = empty_strided_cuda((4, 4, 4, 3), (48, 12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, mul_1, add, mul_2, mul_3, sub, mul_4, mul_5, add_1, mul_6, mul_7, sub_1, mul_8, mul_9, add_2, mul_10, mul_11, sub_2, mul_12, mul_13, add_3, mul_14, mul_15, sub_3], Original ATen: [aten.mul, aten.add, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sub_0.run(buf4, buf6, buf7, buf8, buf14, buf15, buf23, buf24, buf30, buf31, 192, grid=grid(192), stream=stream0)
        del buf2
        del buf3
        del buf4
        del buf5
        del buf6
        buf9 = empty_strided_cuda((4, 4, 4, 3, 2), (96, 24, 6, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [gcc_fft_batch], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_1.run(buf7, buf8, buf9, 384, grid=grid(384), stream=stream0)
        del buf7
        del buf8
        # Topologically Sorted Source Nodes: [gcc_fft_batch_c], Original ATen: [aten.complex]
        buf10 = torch.ops.aten.complex.default(reinterpret_tensor(buf9, (4, 4, 4, 3), (96, 24, 6, 2), 0), reinterpret_tensor(buf9, (4, 4, 4, 3), (96, 24, 6, 2), 1))
        buf11 = buf10
        del buf10
        # Topologically Sorted Source Nodes: [gcc_batch], Original ATen: [aten._fft_c2r]
        buf12 = torch.ops.aten._fft_c2r.default(buf11, [3], 2, 4)
        del buf11
        buf13 = buf12
        del buf12
        buf16 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [gcc_fft_batch_1], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_1.run(buf14, buf15, buf16, 384, grid=grid(384), stream=stream0)
        del buf14
        del buf15
        # Topologically Sorted Source Nodes: [gcc_fft_batch_c_1], Original ATen: [aten.complex]
        buf17 = torch.ops.aten.complex.default(reinterpret_tensor(buf16, (4, 4, 4, 3), (96, 24, 6, 2), 0), reinterpret_tensor(buf16, (4, 4, 4, 3), (96, 24, 6, 2), 1))
        buf18 = buf17
        del buf17
        # Topologically Sorted Source Nodes: [gcc_batch_1], Original ATen: [aten._fft_c2r]
        buf19 = torch.ops.aten._fft_c2r.default(buf18, [3], 2, 4)
        del buf18
        buf20 = buf19
        del buf19
        buf25 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [gcc_fft_batch_2], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_1.run(buf23, buf24, buf25, 384, grid=grid(384), stream=stream0)
        del buf23
        del buf24
        # Topologically Sorted Source Nodes: [gcc_fft_batch_c_2], Original ATen: [aten.complex]
        buf26 = torch.ops.aten.complex.default(reinterpret_tensor(buf25, (4, 4, 4, 3), (96, 24, 6, 2), 0), reinterpret_tensor(buf25, (4, 4, 4, 3), (96, 24, 6, 2), 1))
        buf27 = buf26
        del buf26
        # Topologically Sorted Source Nodes: [gcc_batch_2], Original ATen: [aten._fft_c2r]
        buf28 = torch.ops.aten._fft_c2r.default(buf27, [3], 2, 4)
        del buf27
        buf29 = buf28
        del buf28
        buf32 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [gcc_fft_batch_3], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_1.run(buf30, buf31, buf32, 384, grid=grid(384), stream=stream0)
        del buf30
        del buf31
        # Topologically Sorted Source Nodes: [gcc_fft_batch_c_3], Original ATen: [aten.complex]
        buf33 = torch.ops.aten.complex.default(reinterpret_tensor(buf32, (4, 4, 4, 3), (96, 24, 6, 2), 0), reinterpret_tensor(buf32, (4, 4, 4, 3), (96, 24, 6, 2), 1))
        del buf32
        buf34 = buf33
        del buf33
        # Topologically Sorted Source Nodes: [gcc_batch_3], Original ATen: [aten._fft_c2r]
        buf35 = torch.ops.aten._fft_c2r.default(buf34, [3], 2, 4)
        del buf34
        buf36 = buf35
        del buf35
        buf21 = empty_strided_cuda((4, 4, 4, 5), (80, 20, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_2.run(buf20, buf13, buf0, buf21, 320, grid=grid(320), stream=stream0)
        buf22 = empty_strided_cuda((4, 4, 4, 4, 5), (320, 80, 20, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_3.run(buf21, buf20, buf13, buf0, buf22, 1280, grid=grid(1280), stream=stream0)
        del buf13
        del buf20
        buf37 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [setitem_6], Original ATen: [aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_4.run(buf36, buf29, buf22, buf37, 320, grid=grid(320), stream=stream0)
        buf38 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [setitem_4, setitem_5, setitem_6], Original ATen: [aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_5.run(buf37, buf29, buf22, buf38, 1280, grid=grid(1280), stream=stream0)
        del buf29
        del buf37
        buf39 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_6.run(buf36, buf38, buf39, 1280, grid=grid(1280), stream=stream0)
        del buf36
        del buf38
    return (buf39, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
