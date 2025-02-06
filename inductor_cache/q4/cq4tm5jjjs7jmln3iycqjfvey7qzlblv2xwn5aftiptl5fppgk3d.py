# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/ae/cae3shigxzpbq6yfhn77o6cvodg56sedcmqbtxjyxk773asr4vzh.py
# Topologically Sorted Source Nodes: [x_q, add, input_1], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   input_1 => var_mean
#   x_q => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%primals_1, [1], True), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 0), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused_add_mean_native_layer_norm_0 = async_compile.triton('triton_poi_fused_add_mean_native_layer_norm_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tmp8 + tmp9
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 / tmp7
    tmp19 = tmp18 + tmp9
    tmp20 = tmp10 + tmp19
    tmp23 = tmp21 + tmp22
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp27 / tmp7
    tmp29 = tmp28 + tmp9
    tmp30 = tmp20 + tmp29
    tmp33 = tmp31 + tmp32
    tmp35 = tmp33 + tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37 / tmp7
    tmp39 = tmp38 + tmp9
    tmp40 = tmp30 + tmp39
    tmp41 = tmp40 / tmp7
    tmp42 = tmp10 - tmp41
    tmp43 = tmp42 * tmp42
    tmp44 = tmp19 - tmp41
    tmp45 = tmp44 * tmp44
    tmp46 = tmp43 + tmp45
    tmp47 = tmp29 - tmp41
    tmp48 = tmp47 * tmp47
    tmp49 = tmp46 + tmp48
    tmp50 = tmp39 - tmp41
    tmp51 = tmp50 * tmp50
    tmp52 = tmp49 + tmp51
    tmp53 = tmp52 / tmp7
    tl.store(out_ptr0 + (x0), tmp41, xmask)
    tl.store(out_ptr1 + (x0), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ul/culqhv2bwhd377b4xo64sdsw3wxyg4johrxoitdn2bwdedkgxo3u.py
# Topologically Sorted Source Nodes: [add_1, input_2, input_3], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_3
#   input_2 => add_4, rsqrt_1, var_mean_1
#   input_3 => add_6, rsqrt_2, var_mean_2
# Graph fragment:
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, 0), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
triton_poi_fused_add_native_layer_norm_1 = async_compile.triton('triton_poi_fused_add_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 + tmp1
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 + tmp1
    tmp11 = tmp8 + tmp10
    tmp12 = 4.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp2 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tmp4 - tmp13
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = tmp7 - tmp13
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp10 - tmp13
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24 / tmp12
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp0 + tmp3
    tmp30 = tmp29 + tmp6
    tmp31 = tmp30 + tmp9
    tmp32 = tmp31 / tmp12
    tmp33 = tmp0 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tmp3 - tmp32
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 + tmp36
    tmp38 = tmp6 - tmp32
    tmp39 = tmp38 * tmp38
    tmp40 = tmp37 + tmp39
    tmp41 = tmp9 - tmp32
    tmp42 = tmp41 * tmp41
    tmp43 = tmp40 + tmp42
    tmp44 = tmp43 / tmp12
    tmp45 = tmp44 + tmp26
    tmp46 = libdevice.rsqrt(tmp45)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
    tl.store(out_ptr2 + (x0), tmp32, xmask)
    tl.store(out_ptr3 + (x0), tmp46, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jy/cjyd7ljo6clgcemcdpuwspj6x235rilym6kfx5cx4xikbtkxebmq.py
# Topologically Sorted Source Nodes: [x_q, add, input_1], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   input_1 => add_1, add_2, mul, mul_1, rsqrt, sub
#   x_q => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%primals_1, [1], True), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_3), kwargs = {})
triton_poi_fused_add_mean_native_layer_norm_2 = async_compile.triton('triton_poi_fused_add_mean_native_layer_norm_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (4 + x0 + 16*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (8 + x0 + 16*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (12 + x0 + 16*x1), xmask)
    tmp11 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp12 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qb/cqb72oo467nmkqajmjrrd67edfmjfwizksysxlsy64xul7hj6dh3.py
# Topologically Sorted Source Nodes: [add_1, input_2, input_3], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_3
#   input_2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   input_3 => add_6, add_7, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, 0), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_4), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_5), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %getitem_5), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %primals_6), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %primals_7), kwargs = {})
triton_poi_fused_add_native_layer_norm_3 = async_compile.triton('triton_poi_fused_add_native_layer_norm_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp0 - tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tl.store(out_ptr0 + (x2), tmp10, xmask)
    tl.store(out_ptr1 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/je/cjexbzlyu3wmfsrlb4tx4rc7d5ffqxwcg7nhzmjxezh45bggoizk.py
# Topologically Sorted Source Nodes: [q_2], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   q_2 => mul_6
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 1.0), kwargs = {})
triton_poi_fused_mul_4 = async_compile.triton('triton_poi_fused_mul_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_4(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qv/cqvvputffj3l6llgfyakx7knd4rkyc2cnhaw6be6curiqquqito6.py
# Topologically Sorted Source Nodes: [attn], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/32/c32twtkmrrmzwnbjltfcmg5jpsekciclu3qlquuukyu5qptnog6z.py
# Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => amax, exp, sub_3
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_11, [-1], True), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_11, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
triton_poi_fused__softmax_6 = async_compile.triton('triton_poi_fused__softmax_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
''', device_str='cuda')


# kernel path: inductor_cache/hr/chr4v6m5zsehd4v6276kpjdtw44yi7cjb6ehrrox5sgltgiaxix2.py
# Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_7 = async_compile.triton('triton_poi_fused__softmax_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4), (4, 1))
    assert_size_stride(primals_9, (4, 4), (4, 1))
    assert_size_stride(primals_10, (4, 4), (4, 1))
    assert_size_stride(primals_11, (4, 4), (4, 1))
    assert_size_stride(primals_12, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        buf1 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [x_q, add, input_1], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_native_layer_norm_0.run(primals_1, buf0, buf1, 4, grid=grid(4), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf3 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf4 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf5 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [add_1, input_2, input_3], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_1.run(primals_1, buf2, buf3, buf4, buf5, 16, grid=grid(16), stream=stream0)
        buf6 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_q, add, input_1], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_native_layer_norm_2.run(primals_1, buf0, buf1, primals_2, primals_3, buf6, 16, grid=grid(16), stream=stream0)
        del buf0
        del buf1
        del primals_2
        del primals_3
        buf7 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (4, 4), (4, 1), 0), reinterpret_tensor(primals_8, (4, 4), (1, 4), 0), out=buf7)
        buf8 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf10 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_1, input_2, input_3], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_3.run(primals_1, buf2, buf3, primals_4, primals_5, buf4, buf5, primals_6, primals_7, buf8, buf10, 64, grid=grid(64), stream=stream0)
        del buf2
        del buf3
        del primals_4
        del primals_5
        del primals_6
        del primals_7
        buf9 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (16, 4), (4, 1), 0), reinterpret_tensor(primals_9, (4, 4), (1, 4), 0), out=buf9)
        buf11 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (16, 4), (4, 1), 0), reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), out=buf11)
        buf12 = reinterpret_tensor(buf7, (4, 4, 1, 1), (4, 1, 16, 16), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [q_2], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_4.run(buf12, 16, grid=grid(16), stream=stream0)
        buf13 = empty_strided_cuda((4, 4, 1, 4), (16, 4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf9, buf13, 16, 4, grid=grid(16, 4), stream=stream0)
        buf14 = reinterpret_tensor(buf9, (16, 1, 4), (4, 4, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [attn], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (16, 1, 1), (1, 0, 0), 0), reinterpret_tensor(buf13, (16, 1, 4), (4, 0, 1), 0), out=buf14)
        buf15 = empty_strided_cuda((4, 4, 1, 4), (16, 4, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_6.run(buf14, buf15, 64, grid=grid(64), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (4, 4, 1, 4), (16, 4, 4, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_7.run(buf15, buf16, 64, grid=grid(64), stream=stream0)
        buf17 = reinterpret_tensor(buf15, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf11, buf17, 16, 4, grid=grid(16, 4), stream=stream0)
        del buf11
        buf18 = reinterpret_tensor(buf5, (16, 1, 1), (1, 1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (16, 1, 4), (4, 4, 1), 0), reinterpret_tensor(buf17, (16, 4, 1), (4, 1, 0), 0), out=buf18)
        buf19 = reinterpret_tensor(buf4, (4, 4), (4, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, reinterpret_tensor(buf18, (4, 4), (4, 1), 0), reinterpret_tensor(primals_11, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf19)
        del primals_12
    return (buf19, primals_1, reinterpret_tensor(buf6, (4, 4), (4, 1), 0), reinterpret_tensor(buf8, (16, 4), (4, 1), 0), reinterpret_tensor(buf10, (16, 4), (4, 1), 0), buf16, reinterpret_tensor(buf18, (4, 4), (4, 1), 0), primals_11, reinterpret_tensor(buf17, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf12, (16, 1, 1), (1, 1, 4), 0), reinterpret_tensor(buf13, (16, 4, 1), (4, 1, 4), 0), primals_10, primals_9, primals_8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
