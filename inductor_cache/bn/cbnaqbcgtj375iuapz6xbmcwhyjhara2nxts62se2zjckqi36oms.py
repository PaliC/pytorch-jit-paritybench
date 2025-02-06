# AOT ID: ['2_inference']
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


# kernel path: inductor_cache/py/cpy3dtspa76vnxf7yr2srtqup3sgm6ki7rxft3ta6itk34x3l5rq.py
# Topologically Sorted Source Nodes: [hsv], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   hsv => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%div, %sub_3, %getitem], 1), kwargs = {})
triton_poi_fused_stack_0 = async_compile.triton('triton_poi_fused_stack_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 12)
    x0 = (xindex % 4)
    x2 = xindex // 48
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16 + x0 + 4*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr0 + (32 + x0 + 4*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = 1.7320508075688772
    tmp9 = tmp7 * tmp8
    tmp10 = tl.load(in_ptr0 + (x0 + 4*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp11 = 2.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 - tmp5
    tmp14 = tmp13 - tmp6
    tmp15 = libdevice.atan2(tmp9, tmp14)
    tmp16 = 6.283185307179586
    tmp17 = tmp15 % tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tmp17 != tmp18
    tmp20 = (libdevice.signbit(tmp17) != 0) if (tmp17).dtype is tl.float32 else tmp17 < 0
    tmp21 = (libdevice.signbit(tmp16) != 0) if (tmp16).dtype is tl.float32 else tmp16 < 0
    tmp22 = tmp20 != tmp21
    tmp23 = tmp19 & tmp22
    tmp24 = tmp17 + tmp16
    tmp25 = tl.where(tmp23, tmp24, tmp17)
    tmp26 = 0.15915494309189535
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp4, tmp27, tmp28)
    tmp30 = tmp0 >= tmp3
    tmp31 = tl.full([1], 8, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr0 + (x0 + 4*((-4) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp35 = tl.load(in_ptr0 + (16 + x0 + 4*((-4) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp36 = triton_helpers.minimum(tmp34, tmp35)
    tmp37 = tl.load(in_ptr0 + (32 + x0 + 4*((-4) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp38 = triton_helpers.minimum(tmp36, tmp37)
    tmp39 = tl.load(in_ptr0 + (48 + x0 + 4*((-4) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp40 = triton_helpers.minimum(tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp34, tmp35)
    tmp42 = triton_helpers.maximum(tmp41, tmp37)
    tmp43 = triton_helpers.maximum(tmp42, tmp39)
    tmp44 = 1e-08
    tmp45 = tmp43 + tmp44
    tmp46 = tmp40 / tmp45
    tmp47 = 1.0
    tmp48 = tmp47 - tmp46
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp33, tmp48, tmp49)
    tmp51 = tmp0 >= tmp31
    tmp52 = tl.full([1], 12, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tl.load(in_ptr0 + (x0 + 4*((-8) + x1) + 64*x2), tmp51 & xmask, other=0.0)
    tmp55 = tl.load(in_ptr0 + (16 + x0 + 4*((-8) + x1) + 64*x2), tmp51 & xmask, other=0.0)
    tmp56 = triton_helpers.maximum(tmp54, tmp55)
    tmp57 = tl.load(in_ptr0 + (32 + x0 + 4*((-8) + x1) + 64*x2), tmp51 & xmask, other=0.0)
    tmp58 = triton_helpers.maximum(tmp56, tmp57)
    tmp59 = tl.load(in_ptr0 + (48 + x0 + 4*((-8) + x1) + 64*x2), tmp51 & xmask, other=0.0)
    tmp60 = triton_helpers.maximum(tmp58, tmp59)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp51, tmp60, tmp61)
    tmp63 = tl.where(tmp33, tmp50, tmp62)
    tmp64 = tl.where(tmp4, tmp29, tmp63)
    tl.store(out_ptr0 + (x3), tmp64, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j7/cj7u43oa4yv6p2tuxptdxpcxtmzcwhvz4n5c6sl56bczoxy44adp.py
# Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem => full_default_3, index_put
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : [num_users=4] = call_function[target=torch.ops.aten.index_put_.default](args = (%view, [%bitwise_not], %full_default_3), kwargs = {})
triton_poi_fused_index_put_lift_fresh_1 = async_compile.triton('triton_poi_fused_index_put_lift_fresh_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_1', 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_put_lift_fresh_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0 == tmp0
    tmp2 = tl_math.abs(tmp0)
    tmp3 = float("inf")
    tmp4 = tmp2 != tmp3
    tmp5 = tmp1 & tmp4
    tmp6 = tmp5 == 0
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, tmp0)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tv/ctvunf4uqlju25jjptesxeiz5odzssjmfsot2iow6zmolgzkurr5.py
# Topologically Sorted Source Nodes: [f_h], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   f_h => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 1, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_new_zeros_2 = async_compile.triton('triton_poi_fused_new_zeros_2', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gf/cgfpd23qcabtdu5rn5jxqya3sjqodazferiphbfnmhleoa746sxn.py
# Topologically Sorted Source Nodes: [f_s], Original ATen: [aten.new_ones]
# Source node to ATen node mapping:
#   f_s => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 1, 1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_new_ones_3 = async_compile.triton('triton_poi_fused_new_ones_3', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_ones_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_ones_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/am/camsnhk6isxhygljpapc73c7wlclfy6kluytyyrpzk6hnwgt33kt.py
# Topologically Sorted Source Nodes: [mul_2, truediv_2, h_1, h_2, setitem_1, mul_3, setitem_2], Original ATen: [aten.mul, aten.div, aten.add, aten.remainder, aten.copy]
# Source node to ATen node mapping:
#   h_1 => add_1
#   h_2 => remainder_1
#   mul_2 => mul_3
#   mul_3 => mul_4
#   setitem_1 => copy
#   setitem_2 => copy_1
#   truediv_2 => div_2
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%uniform, 255.0), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, 360.0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_4, %div_2), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int, %add_1, 2, 0, 9223372036854775807), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put, %slice_scatter_default, 1, 0), kwargs = {})
#   %remainder_1 : [num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%select_6, 1), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_8, %remainder_1), kwargs = {})
#   %select_scatter_default_1 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %copy, 1, 0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %uniform_1), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_14, %mul_4), kwargs = {})
#   %select_scatter_default_2 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %copy_1, 1, 1), kwargs = {})
triton_poi_fused_add_copy_div_mul_remainder_4 = async_compile.triton('triton_poi_fused_add_copy_div_mul_remainder_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_remainder_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy_div_mul_remainder_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 3)
    x0 = (xindex % 16)
    x2 = xindex // 48
    x3 = xindex
    tmp6 = tl.load(in_ptr0 + (x0 + 48*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (16 + x0 + 48*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (x3), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp5 = tmp3 == tmp3
    tmp8 = 255.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.002777777777777778
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.where(tmp5, tmp12, tmp6)
    tmp14 = 1.0
    tmp15 = tmp13 % tmp14
    tmp16 = tmp15 != tmp3
    tmp17 = (libdevice.signbit(tmp15) != 0) if (tmp15).dtype is tl.float32 else tmp15 < 0
    tmp18 = (libdevice.signbit(tmp14) != 0) if (tmp14).dtype is tl.float32 else tmp14 < 0
    tmp19 = tmp17 != tmp18
    tmp20 = tmp16 & tmp19
    tmp21 = tmp15 + tmp14
    tmp22 = tl.where(tmp20, tmp21, tmp15)
    tmp24 = tl.where(tmp4, tmp12, tmp23)
    tmp25 = tl.where(tmp4, tmp22, tmp24)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp0 == tmp3
    tmp30 = tl.where(tmp28, tmp12, tmp29)
    tmp31 = tl.where(tmp28, tmp22, tmp30)
    tmp32 = tl.where(tmp2, tmp27, tmp31)
    tl.store(out_ptr0 + (x3), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/py/cpyhixfqzvowd5vkrt6flmxnltramy3zjpbvhfwlhemqtqzmkl2g.py
# Topologically Sorted Source Nodes: [mul_4, setitem_3, x, v, s, c, h_3, mul_6, add_1, k, sub_4, t, t_1, mul_7, x_1, means, sub_6, mul_8, x_2, inputs], Original ATen: [aten.mul, aten.copy, aten.clamp, aten.index, aten.add, aten.remainder, aten.rsub, aten.minimum, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   add_1 => add_2
#   c => mul_6
#   h_3 => index
#   inputs => clamp_max_2, clamp_min_2
#   k => remainder_2
#   means => mean
#   mul_4 => mul_5
#   mul_6 => mul_7
#   mul_7 => mul_8
#   mul_8 => mul_9
#   s => index_1
#   setitem_3 => copy_2
#   sub_4 => sub_4
#   sub_6 => sub_6
#   t => minimum
#   t_1 => clamp_max_1, clamp_min_1
#   v => index_2
#   x => clamp_max, clamp_min
#   x_1 => sub_5
#   x_2 => add_3
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %uniform_2), kwargs = {})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_20, %mul_5), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_2, %copy_2, 1, 2), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%select_scatter_default_3, 0), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1), kwargs = {})
#   %index_2 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%clamp_max, [None, %full_default_6]), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%clamp_max, [None, %full_default_5]), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_2, %index_1), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%clamp_max, [None, %full_default_4]), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, 6), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %mul_7), kwargs = {})
#   %remainder_2 : [num_users=2] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%add_2, 6), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (4.0, %remainder_2), kwargs = {})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%remainder_2, %sub_4), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%minimum, 0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %clamp_max_1), kwargs = {})
#   %sub_5 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index_2, %mul_8), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_5, [2, 3], True), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_5, %mean), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %uniform_3), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mean), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0), kwargs = {})
#   %clamp_max_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1), kwargs = {})
triton_per_fused_add_clamp_copy_index_mean_minimum_mul_remainder_rsub_sub_5 = async_compile.triton('triton_per_fused_add_clamp_copy_index_mean_minimum_mul_remainder_rsub_sub_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_copy_index_mean_minimum_mul_remainder_rsub_sub_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_copy_index_mean_minimum_mul_remainder_rsub_sub_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = (xindex % 3)
    r2 = rindex
    x1 = xindex // 3
    x3 = xindex
    tmp13 = tl.load(in_ptr0 + (32 + r2 + 48*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (r2 + 48*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr0 + (16 + r2 + 48*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1, 1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 3.0
    tmp6 = 1.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 5.0
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp10 = tl.full([1, 1], 0, tl.int32)
    tmp11 = tl.full([1, 1], 2, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp15 = tmp13 * tmp14
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = triton_helpers.minimum(tmp19, tmp6)
    tmp21 = 6.0
    tmp22 = tmp20 * tmp21
    tmp23 = tmp9 + tmp22
    tmp24 = tmp23 % tmp21
    tmp25 = tmp24 != tmp10
    tmp26 = (libdevice.signbit(tmp24) != 0) if (tmp24).dtype is tl.float32 else tmp24 < 0
    tmp27 = (libdevice.signbit(tmp21) != 0) if (tmp21).dtype is tl.float32 else tmp21 < 0
    tmp28 = tmp26 != tmp27
    tmp29 = tmp25 & tmp28
    tmp30 = tmp24 + tmp21
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp32 = 4.0
    tmp33 = tmp32 - tmp31
    tmp34 = triton_helpers.minimum(tmp31, tmp33)
    tmp35 = triton_helpers.maximum(tmp34, tmp18)
    tmp36 = tmp11 == tmp11
    tmp37 = tl.where(tmp36, tmp15, tmp13)
    tmp38 = triton_helpers.maximum(tmp37, tmp18)
    tmp39 = triton_helpers.minimum(tmp38, tmp6)
    tmp40 = tl.full([1, 1], 1, tl.int32)
    tmp41 = tmp40 == tmp11
    tmp43 = tl.where(tmp41, tmp15, tmp42)
    tmp44 = triton_helpers.maximum(tmp43, tmp18)
    tmp45 = triton_helpers.minimum(tmp44, tmp6)
    tmp46 = tmp39 * tmp45
    tmp47 = triton_helpers.minimum(tmp35, tmp6)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp39 - tmp48
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
    tmp52 = tl.where(xmask, tmp50, 0)
    tmp53 = tl.sum(tmp52, 1)[:, None]
    tmp54 = 16.0
    tmp55 = tmp53 / tmp54
    tmp56 = tmp49 - tmp55
    tmp58 = tmp56 * tmp57
    tmp59 = tmp58 + tmp55
    tmp60 = triton_helpers.maximum(tmp59, tmp18)
    tmp61 = triton_helpers.minimum(tmp60, tmp6)
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp61, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 12, 4), (48, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hsv], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_0.run(arg0_1, buf0, 192, grid=grid(192), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_put_lift_fresh_1.run(buf0, buf0, 192, grid=grid(192), stream=stream0)
        buf7 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [f_h], Original ATen: [aten.new_zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_zeros_2.run(buf7, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [f_h, uniform_], Original ATen: [aten.new_zeros, aten.uniform]
        buf8 = torch.ops.aten.uniform.default(buf7, -4.0, 4.0)
        buf9 = buf8
        del buf8
        buf10 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [f_s], Original ATen: [aten.new_ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_ones_3.run(buf10, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [f_s, f_s_1], Original ATen: [aten.new_ones, aten.uniform]
        buf11 = torch.ops.aten.uniform.default(buf10, 0.0, 5.0)
        del buf10
        buf12 = buf11
        del buf11
        buf13 = empty_strided_cuda((4, 3, 4, 4), (48, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_2, truediv_2, h_1, h_2, setitem_1, mul_3, setitem_2], Original ATen: [aten.mul, aten.div, aten.add, aten.remainder, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy_div_mul_remainder_4.run(buf0, buf9, buf12, buf13, 192, grid=grid(192), stream=stream0)
        del buf12
        buf14 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [f_v], Original ATen: [aten.new_ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_ones_3.run(buf14, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [f_v, f_v_1], Original ATen: [aten.new_ones, aten.uniform]
        buf15 = torch.ops.aten.uniform.default(buf14, 0.0, 5.0)
        buf16 = buf15
        del buf15
        buf20 = reinterpret_tensor(buf14, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [factor], Original ATen: [aten.uniform]
        buf21 = torch.ops.aten.uniform.default(buf20, 0.0, 5.0)
        del buf20
        buf22 = buf21
        del buf21
        buf17 = reinterpret_tensor(buf0, (4, 3, 4, 4), (48, 16, 4, 1), 0); del buf0  # reuse
        buf18 = buf17; del buf17  # reuse
        buf23 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [mul_4, setitem_3, x, v, s, c, h_3, mul_6, add_1, k, sub_4, t, t_1, mul_7, x_1, means, sub_6, mul_8, x_2, inputs], Original ATen: [aten.mul, aten.copy, aten.clamp, aten.index, aten.add, aten.remainder, aten.rsub, aten.minimum, aten.sub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_copy_index_mean_minimum_mul_remainder_rsub_sub_5.run(buf23, buf13, buf16, buf22, 12, 16, grid=grid(12), stream=stream0)
        del buf13
        del buf16
        del buf22
    return (buf23, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
