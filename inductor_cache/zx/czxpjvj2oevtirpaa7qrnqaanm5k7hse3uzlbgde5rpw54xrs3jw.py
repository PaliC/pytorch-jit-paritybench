# AOT ID: ['9_forward']
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


# kernel path: inductor_cache/yc/cyc3alpuaetjs733jm7t7w2nk55igron4uqko7tu3aaqh2im2k46.py
# Topologically Sorted Source Nodes: [truediv_2, sub_1, sign, abs_3, add_2, floor, output, clamp, add_3, output_1], Original ATen: [aten.div, aten.sub, aten.sign, aten.abs, aten.add, aten.floor, aten.mul, aten.clamp]
# Source node to ATen node mapping:
#   abs_3 => abs_3
#   add_2 => add_2
#   add_3 => add_3
#   clamp => clamp_max, clamp_min
#   floor => floor
#   output => mul_2
#   output_1 => mul_3
#   sign => sign
#   sub_1 => sub_1
#   truediv_2 => div_2
# Graph fragment:
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_7, %primals_6), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_2, %primals_8), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub_1,), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_3, 0.5), kwargs = {})
#   %floor : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%add_2,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %floor), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.Tensor](args = (%mul_2, %primals_11), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.Tensor](args = (%clamp_min, %primals_12), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max, %primals_8), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %primals_6), kwargs = {})
triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0 = async_compile.triton('triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp19 = tl.load(in_ptr3 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp22 = tl.load(in_ptr4 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tmp6 = tmp3 - tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp7 < tmp6
    tmp9 = tmp8.to(tl.int8)
    tmp10 = tmp6 < tmp7
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp9 - tmp11
    tmp13 = tmp12.to(tmp6.dtype)
    tmp14 = tl_math.abs(tmp6)
    tmp15 = 0.5
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.floor(tmp16)
    tmp18 = tmp13 * tmp17
    tmp21 = triton_helpers.maximum(tmp18, tmp20)
    tmp24 = triton_helpers.minimum(tmp21, tmp23)
    tmp25 = tmp24 + tmp5
    tmp26 = tmp25 * tmp2
    tl.store(out_ptr0 + (x0), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x2/cx2yxrf6efmsyouytlkmouqqfvle5w2mzbkhn74gokjaqubert5v.py
# Topologically Sorted Source Nodes: [weight_fused, truediv_5, sub_4, output_2, clamp_1, add_4, output_3], Original ATen: [aten.mul, aten.div, aten.sub, aten.sign, aten.abs, aten.add, aten.floor, aten.clamp]
# Source node to ATen node mapping:
#   add_4 => add_5
#   clamp_1 => clamp_max_1, clamp_min_1
#   output_2 => abs_6, add_4, floor_1, mul_4, sign_1
#   output_3 => mul_5
#   sub_4 => sub_4
#   truediv_5 => div_5
#   weight_fused => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %view_1), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_1, %primals_13), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_5, %primals_14), kwargs = {})
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub_4,), kwargs = {})
#   %abs_6 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_4,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_6, 0.5), kwargs = {})
#   %floor_1 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%add_4,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, %floor_1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.Tensor](args = (%mul_4, %primals_17), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.Tensor](args = (%clamp_min_1, %primals_18), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_1, %primals_14), kwargs = {})
#   %mul_5 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %primals_13), kwargs = {})
triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_1 = async_compile.triton('triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp27 = tl.load(in_ptr6 + (0))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp3 = 1e-05
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.sqrt(tmp4)
    tmp6 = tmp1 / tmp5
    tmp7 = tmp0 * tmp6
    tmp9 = tmp7 / tmp8
    tmp11 = tmp9 - tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = tmp12 < tmp11
    tmp14 = tmp13.to(tl.int8)
    tmp15 = tmp11 < tmp12
    tmp16 = tmp15.to(tl.int8)
    tmp17 = tmp14 - tmp16
    tmp18 = tmp17.to(tmp11.dtype)
    tmp19 = tl_math.abs(tmp11)
    tmp20 = 0.5
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.floor(tmp21)
    tmp23 = tmp18 * tmp22
    tmp26 = triton_helpers.maximum(tmp23, tmp25)
    tmp29 = triton_helpers.minimum(tmp26, tmp28)
    tmp30 = tmp29 + tmp10
    tmp31 = tmp30 * tmp8
    tl.store(in_out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6y/c6yd4uay4o5vp7c2h47v6mmg7l6uyroqxveerbitfd4raz2flalx.py
# Topologically Sorted Source Nodes: [truediv_6, sub_5, truediv_7, sub_6, output_2], Original ATen: [aten.div, aten.sub, aten.abs, aten.maximum]
# Source node to ATen node mapping:
#   output_2 => abs_4, abs_5, maximum_1
#   sub_5 => sub_5
#   sub_6 => sub_6
#   truediv_6 => div_6
#   truediv_7 => div_7
# Graph fragment:
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_15, %primals_13), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_6, %primals_14), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_16, %primals_13), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_7, %primals_14), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_5,), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_6,), kwargs = {})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%abs_4, %abs_5), kwargs = {})
triton_poi_fused_abs_div_maximum_sub_2 = async_compile.triton('triton_poi_fused_abs_div_maximum_sub_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_div_maximum_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_div_maximum_sub_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = tl_math.abs(tmp4)
    tmp7 = tmp6 / tmp1
    tmp8 = tmp7 - tmp3
    tmp9 = tl_math.abs(tmp8)
    tmp10 = triton_helpers.maximum(tmp5, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nm/cnmtx2c4l5f3gxmib4jgjxilsywalivmcphwtx6e4dxuhrpdesj6.py
# Topologically Sorted Source Nodes: [add, sqrt, truediv, mul, sub, bias_fused, output_4], Original ATen: [aten.add, aten.sqrt, aten.div, aten.mul, aten.sub, aten.view, aten.convolution]
# Source node to ATen node mapping:
#   add => add
#   bias_fused => view
#   mul => mul
#   output_4 => convolution
#   sqrt => sqrt
#   sub => sub
#   truediv => div
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, 1e-05), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_3, %sqrt), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %div), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %mul), kwargs = {})
#   %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub, [-1]), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_3, %mul_5, %view, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_3 = async_compile.triton('triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp3 = tl.load(in_ptr3 + (x0), xmask)
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tmp2 / tmp6
    tmp8 = tmp1 * tmp7
    tmp9 = tmp0 - tmp8
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/um/cumrylsbycd2c4tut4nnfhqxynqqy4f6jg3u664g6yrjzg2ygmcj.py
# Topologically Sorted Source Nodes: [add, sqrt, truediv, mul, sub, bias_fused, output_4], Original ATen: [aten.add, aten.sqrt, aten.div, aten.mul, aten.sub, aten.view, aten.convolution]
# Source node to ATen node mapping:
#   add => add
#   bias_fused => view
#   mul => mul
#   output_4 => convolution
#   sqrt => sqrt
#   sub => sub
#   truediv => div
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, 1e-05), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_3, %sqrt), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %div), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %mul), kwargs = {})
#   %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub, [-1]), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_3, %mul_5, %view, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_4 = async_compile.triton('triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18 = args
    args.clear()
    assert_size_stride(primals_1, (4, ), (1, ))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_6, (1, ), (1, ))
    assert_size_stride(primals_7, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_8, (1, ), (1, ))
    assert_size_stride(primals_9, (1, ), (1, ))
    assert_size_stride(primals_10, (1, ), (1, ))
    assert_size_stride(primals_11, (), ())
    assert_size_stride(primals_12, (), ())
    assert_size_stride(primals_13, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_14, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_15, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_16, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_17, (), ())
    assert_size_stride(primals_18, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv_2, sub_1, sign, abs_3, add_2, floor, output, clamp, add_3, output_1], Original ATen: [aten.div, aten.sub, aten.sign, aten.abs, aten.add, aten.floor, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0.run(primals_7, primals_6, primals_8, primals_11, primals_12, buf0, 256, grid=grid(256), stream=stream0)
        del primals_11
        del primals_12
        del primals_6
        del primals_7
        del primals_8
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [weight_fused, truediv_5, sub_4, output_2, clamp_1, add_4, output_3], Original ATen: [aten.mul, aten.div, aten.sub, aten.sign, aten.abs, aten.add, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_1.run(buf3, primals_5, primals_3, primals_4, primals_13, primals_14, primals_17, primals_18, 256, grid=grid(256), stream=stream0)
        buf2 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv_6, sub_5, truediv_7, sub_6, output_2], Original ATen: [aten.div, aten.sub, aten.abs, aten.maximum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_div_maximum_sub_2.run(primals_15, primals_13, primals_14, primals_16, buf2, 4, grid=grid(4), stream=stream0)
        del primals_15
        del primals_16
        buf4 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [add, sqrt, truediv, mul, sub, bias_fused, output_4], Original ATen: [aten.add, aten.sqrt, aten.div, aten.mul, aten.sub, aten.view, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_3.run(primals_1, primals_2, primals_3, primals_4, buf4, 4, grid=grid(4), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [add, sqrt, truediv, mul, sub, bias_fused, output_4], Original ATen: [aten.add, aten.sqrt, aten.div, aten.mul, aten.sub, aten.view, aten.convolution]
        buf5 = extern_kernels.convolution(buf0, buf3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 4, 1, 1), (4, 1, 1, 1))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [add, sqrt, truediv, mul, sub, bias_fused, output_4], Original ATen: [aten.add, aten.sqrt, aten.div, aten.mul, aten.sub, aten.view, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mul_sqrt_sub_view_4.run(buf6, buf4, 16, grid=grid(16), stream=stream0)
        del buf4
    return (buf6, primals_2, primals_3, primals_4, primals_5, primals_13, primals_14, primals_17, primals_18, buf0, buf2, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
