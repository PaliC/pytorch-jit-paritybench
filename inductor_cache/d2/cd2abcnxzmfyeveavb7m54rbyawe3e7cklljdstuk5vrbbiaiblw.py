# AOT ID: ['15_forward']
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


# kernel path: inductor_cache/z3/cz3sgltd6p5uy3lhiej2fjplx5zxmtjfj6icr5tt4fjuzkhdn7yq.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/jk/cjk7cijecz4qsjo7e44q6mozuyy64643ugcn3xjvntpz245t4lca.py
# Topologically Sorted Source Nodes: [x_3, batch_norm_1, relu_1, batch_norm_2, relu_2], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_4, mul_5, sub_1
#   batch_norm_2 => add_5, mul_7, mul_8, sub_2
#   relu_1 => relu_1
#   relu_2 => relu_2
#   x_3 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x6 = xindex // 16
    x7 = xindex
    x4 = ((xindex // 256) % 64)
    tmp77 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr7 + (x4), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr8 + (x4), None, eviction_policy='evict_last')
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-33) + 2*x0 + 64*x6), tmp10, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-32) + 2*x0 + 64*x6), tmp16, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-31) + 2*x0 + 64*x6), tmp23, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 64*x6), tmp30, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 64*x6), tmp33, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x6), tmp36, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (31 + 2*x0 + 64*x6), tmp43, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x6), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x6), tmp49, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp78 = tmp51 - tmp77
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp89 = tmp87 * tmp88
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp95 = tmp51 - tmp94
    tmp97 = tmp96 + tmp80
    tmp98 = libdevice.sqrt(tmp97)
    tmp99 = tmp83 / tmp98
    tmp100 = tmp99 * tmp85
    tmp101 = tmp95 * tmp100
    tmp103 = tmp101 * tmp102
    tmp105 = tmp103 + tmp104
    tmp106 = triton_helpers.maximum(tmp92, tmp105)
    tl.store(out_ptr0 + (x7), tmp51, None)
    tl.store(out_ptr1 + (x7), tmp76, None)
    tl.store(out_ptr2 + (x7), tmp93, None)
    tl.store(out_ptr3 + (x7), tmp106, None)
''', device_str='cuda')


# kernel path: inductor_cache/7z/c7zqund2kyqpvq4ysgdb6xdjvyh2redsgdcb7t7ionmzslvhclsd.py
# Topologically Sorted Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_3 => add_7, mul_10, mul_11, sub_3
#   relu_3 => relu_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 96)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/dn/cdn4gtgezpsryoswtqcx7jhedxyzynug6cgirjevw3endti6kmgs.py
# Topologically Sorted Source Nodes: [x_in_3, batch_norm_5, relu_5], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_5 => add_12, mul_16, mul_17, sub_5
#   relu_5 => relu_5
#   x_in_3 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_10, %cat], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 311296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 304)
    x0 = (xindex % 256)
    x2 = xindex // 77824
    x3 = xindex
    tmp29 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 73728*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 256*(x1) + 69632*x2), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 304, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-256) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 32, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr0 + (65536 + x0 + 256*((-256) + x1) + 73728*x2), tmp18, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 48, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (65536 + x0 + 256*((-32) + ((-256) + x1)) + 69632*x2), tmp23, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/ji/cjiug5swwll4xxmkn4p357n4r2ol7f3qy6xvyyw4k62bwgwjzwpb.py
# Topologically Sorted Source Nodes: [x_in_7, batch_norm_8, relu_8], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_8 => add_19, mul_25, mul_26, sub_8
#   relu_8 => relu_8
#   x_in_7 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_17, %cat_2], 1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_3, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 320)
    x0 = (xindex % 256)
    x2 = xindex // 81920
    x3 = xindex
    tmp45 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 73728*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 256*(x1) + 69632*x2), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 256*(x1) + 69632*x2), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 320, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-256) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 48, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-256) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 32, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr0 + (65536 + x0 + 256*((-256) + x1) + 73728*x2), tmp26, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 48, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (65536 + x0 + 256*((-32) + ((-256) + x1)) + 69632*x2), tmp31, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 64, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (65536 + x0 + 256*((-48) + ((-256) + x1)) + 69632*x2), tmp39, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, None)
    tl.store(out_ptr1 + (x3), tmp61, None)
''', device_str='cuda')


# kernel path: inductor_cache/qe/cqea36zsbtyzacc6swbxqekl67jylnr5qrfrc73jok3f77epyezo.py
# Topologically Sorted Source Nodes: [dense_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_2 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_2, %slice_30], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 80)
    x0 = (xindex % 256)
    x2 = xindex // 20480
    x3 = (xindex % 20480)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 48, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 32, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (65536 + x0 + 256*(x1) + 73728*x2), tmp16, other=0.0)
    tmp18 = tmp11 >= tmp14
    tmp19 = tl.full([1], 48, tl.int64)
    tmp20 = tmp11 < tmp19
    tmp21 = tmp18 & tmp10
    tmp22 = tl.load(in_ptr1 + (65536 + x0 + 256*((-32) + (x1)) + 69632*x2), tmp21, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tmp5 >= tmp8
    tmp27 = tl.full([1], 64, tl.int64)
    tmp28 = tmp5 < tmp27
    tmp29 = tmp26 & tmp4
    tmp30 = tl.load(in_ptr2 + (65536 + x0 + 256*((-48) + (x1)) + 69632*x2), tmp29, other=0.0)
    tmp31 = tl.where(tmp9, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 80, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr3 + (65536 + x0 + 256*((-64) + x1) + 69632*x2), tmp34, other=0.0)
    tmp38 = tl.where(tmp4, tmp33, tmp37)
    tl.store(out_ptr0 + (x3 + 86016*x2), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/pc/cpcjopo4uryukjn4toatyf4givs2jestvx6xb3rxe43eugdc4nhu.py
# Topologically Sorted Source Nodes: [resid, resid_1, resid_2], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid => add_10
#   resid_1 => add_17
#   resid_2 => add_24
# Graph fragment:
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_2, %slice_10), kwargs = {})
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %slice_18), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %slice_26), kwargs = {})
triton_poi_fused_add_6 = async_compile.triton('triton_poi_fused_add_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 65536)
    x1 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + (x0 + 73728*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 69632*x1), None)
    tmp3 = tl.load(in_ptr2 + (x0 + 69632*x1), None)
    tmp5 = tl.load(in_ptr3 + (x0 + 69632*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0 + 86016*x1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/ew/cew2os4c2s3mutetzl6xzeq6bzlhlr4kzfda2kbne7nfd7wanmxc.py
# Topologically Sorted Source Nodes: [batch_norm_11, relu_11, batch_norm_12, relu_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_11 => add_26, mul_34, mul_35, sub_11
#   batch_norm_12 => add_28, mul_37, mul_38, sub_12
#   relu_11 => relu_11
#   relu_12 => relu_12
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_5, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_5, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 344064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 336)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp0 - tmp18
    tmp21 = tmp20 + tmp4
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tmp7 / tmp22
    tmp24 = tmp23 * tmp9
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = triton_helpers.maximum(tmp16, tmp29)
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/uu/cuux63wrnfvlhugiscnmftnokjj44tzzfujc6ysx4dywicv3piqq.py
# Topologically Sorted Source Nodes: [batch_norm_13, relu_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_13 => add_30, mul_40, mul_41, sub_13
#   relu_13 => relu_13
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_30,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 192)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/do/cdo6tnuffdmv6aqltimijj5ok3ymrtuklsj5ipl3a7y2aaguwld7.py
# Topologically Sorted Source Nodes: [batch_norm_14, relu_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_14 => add_32, mul_43, mul_44, sub_14
#   relu_14 => relu_14
# Graph fragment:
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_32,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 192)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/yi/cyiakctxgxx6jyuw3xudfi2i4vnijqfy3vayf7igxfubjeosv7rz.py
# Topologically Sorted Source Nodes: [x_in_15, batch_norm_15, relu_15], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_15 => add_35, mul_46, mul_47, sub_15
#   relu_15 => relu_15
#   x_in_15 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_33, %cat_6], 1), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_7, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_35,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 155648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 608)
    x0 = (xindex % 64)
    x2 = xindex // 38912
    x3 = xindex
    tmp29 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 36864*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 64*(x1) + 34816*x2), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 608, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-512) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 64, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr0 + (32768 + x0 + 64*((-512) + x1) + 36864*x2), tmp18, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 96, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (32768 + x0 + 64*((-64) + ((-512) + x1)) + 34816*x2), tmp23, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/6b/c6bgmbtsvergv2qalkj65jwft4r6iclevjlizhux7srgusi33m3g.py
# Topologically Sorted Source Nodes: [x_in_19, batch_norm_18, relu_18], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_18 => add_42, mul_55, mul_56, sub_18
#   relu_18 => relu_18
#   x_in_19 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_40, %cat_8], 1), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_9, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %relu_18 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_42,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 640)
    x0 = (xindex % 64)
    x2 = xindex // 40960
    x3 = xindex
    tmp45 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 36864*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 64*(x1) + 34816*x2), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 64*(x1) + 34816*x2), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 640, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-512) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 96, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-512) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 64, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr0 + (32768 + x0 + 64*((-512) + x1) + 36864*x2), tmp26, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 96, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (32768 + x0 + 64*((-64) + ((-512) + x1)) + 34816*x2), tmp31, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 128, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (32768 + x0 + 64*((-96) + ((-512) + x1)) + 34816*x2), tmp39, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, None)
    tl.store(out_ptr1 + (x3), tmp61, None)
''', device_str='cuda')


# kernel path: inductor_cache/mx/cmxtr6m25ql3wc7moexsju7pbe5m36yajrp5a7ajt2vww5ha2o5z.py
# Topologically Sorted Source Nodes: [dense_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_5 => cat_10
# Graph fragment:
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_8, %slice_62], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 160)
    x0 = (xindex % 64)
    x2 = xindex // 10240
    x3 = (xindex % 10240)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 96, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 64, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (32768 + x0 + 64*(x1) + 36864*x2), tmp16, other=0.0)
    tmp18 = tmp11 >= tmp14
    tmp19 = tl.full([1], 96, tl.int64)
    tmp20 = tmp11 < tmp19
    tmp21 = tmp18 & tmp10
    tmp22 = tl.load(in_ptr1 + (32768 + x0 + 64*((-64) + (x1)) + 34816*x2), tmp21, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tmp5 >= tmp8
    tmp27 = tl.full([1], 128, tl.int64)
    tmp28 = tmp5 < tmp27
    tmp29 = tmp26 & tmp4
    tmp30 = tl.load(in_ptr2 + (32768 + x0 + 64*((-96) + (x1)) + 34816*x2), tmp29, other=0.0)
    tmp31 = tl.where(tmp9, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 160, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr3 + (32768 + x0 + 64*((-128) + x1) + 34816*x2), tmp34, other=0.0)
    tmp38 = tl.where(tmp4, tmp33, tmp37)
    tl.store(out_ptr0 + (x3 + 43008*x2), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/ns/cnspo7brmtyueiydfdvszrzz6jm75wdnmwnqwyie5adkbl753q6y.py
# Topologically Sorted Source Nodes: [resid_3, resid_4, resid_5], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_3 => add_33
#   resid_4 => add_40
#   resid_5 => add_47
# Graph fragment:
#   %add_33 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_34, %slice_42), kwargs = {})
#   %add_40 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %slice_50), kwargs = {})
#   %add_47 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %slice_58), kwargs = {})
triton_poi_fused_add_13 = async_compile.triton('triton_poi_fused_add_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32768)
    x1 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + (x0 + 36864*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 34816*x1), None)
    tmp3 = tl.load(in_ptr2 + (x0 + 34816*x1), None)
    tmp5 = tl.load(in_ptr3 + (x0 + 34816*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0 + 43008*x1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/d2/cd25cgzxgjnv5qu7aje5di6zetaphajrxmfkrkisz233owhbzpbo.py
# Topologically Sorted Source Nodes: [batch_norm_21, relu_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_21 => add_49, mul_64, mul_65, sub_21
#   relu_21 => relu_21
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_11, %unsqueeze_169), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_173), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_175), kwargs = {})
#   %relu_21 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_49,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 172032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 672)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/na/cnauqvhdxeuj3fq5dhpejkea2xgqh3qyic4oilnhzbjimnkrjwxk.py
# Topologically Sorted Source Nodes: [x_in_27, batch_norm_24, relu_24, batch_norm_25, relu_25], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_24 => add_56, mul_73, mul_74, sub_24
#   batch_norm_25 => add_58, mul_76, mul_77, sub_25
#   relu_24 => relu_24
#   relu_25 => relu_25
#   x_in_27 => cat_13
# Graph fragment:
#   %cat_13 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_54, %cat_12], 1), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_13, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_13, %unsqueeze_201), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %relu_25 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_58,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 180224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 704)
    x0 = (xindex % 64)
    x2 = xindex // 45056
    x3 = xindex
    tmp29 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 43008*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 64*(x1) + 34816*x2), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 704, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-512) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 160, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr2 + (x0 + 64*((-512) + x1) + 43008*x2), tmp18, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 192, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (32768 + x0 + 64*((-160) + ((-512) + x1)) + 34816*x2), tmp23, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp47 = tmp28 - tmp46
    tmp49 = tmp48 + tmp32
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tmp35 / tmp50
    tmp52 = tmp51 * tmp37
    tmp53 = tmp47 * tmp52
    tmp55 = tmp53 * tmp54
    tmp57 = tmp55 + tmp56
    tmp58 = triton_helpers.maximum(tmp44, tmp57)
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp45, None)
    tl.store(out_ptr2 + (x3), tmp58, None)
''', device_str='cuda')


# kernel path: inductor_cache/sk/cskir4qinxim2blxgxcrylvjnki37rrlg4qpiqwhloycjj7n7ww4.py
# Topologically Sorted Source Nodes: [batch_norm_26, relu_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_26 => add_60, mul_79, mul_80, sub_26
#   relu_26 => relu_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %relu_26 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_60,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 384)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crzalodsatb4lazir3ldl7ounsuedjzu22pbq3b73vyunmfbzux7.py
# Topologically Sorted Source Nodes: [batch_norm_27, relu_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_27 => add_62, mul_82, mul_83, sub_27
#   relu_27 => relu_27
# Graph fragment:
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_217), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_221), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_223), kwargs = {})
#   %relu_27 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_62,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 384)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/xj/cxjxqjo2cmniviwpipc5dilbebhvvl2lyeisjsappgnqmhb4aegt.py
# Topologically Sorted Source Nodes: [x_in_31, batch_norm_28, relu_28], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_28 => add_65, mul_85, mul_86, sub_28
#   relu_28 => relu_28
#   x_in_31 => cat_15
# Graph fragment:
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_63, %cat_14], 1), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_15, %unsqueeze_225), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %relu_28 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 70144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1096)
    x0 = (xindex % 16)
    x2 = xindex // 17536
    x3 = xindex
    tmp29 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 17152*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1096, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 48, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr0 + (16384 + x0 + 16*((-1024) + x1) + 17152*x2), tmp18 & xmask, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 72, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (16384 + x0 + 16*((-48) + ((-1024) + x1)) + 16768*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gu/cguf5w7z6xjs3hum7cz7pj45opnqm4flso455gjlk6zqmoskhryd.py
# Topologically Sorted Source Nodes: [x_in_35, batch_norm_31, relu_31], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_31 => add_72, mul_94, mul_95, sub_31
#   relu_31 => relu_31
#   x_in_35 => cat_17
# Graph fragment:
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_70, %cat_16], 1), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_17, %unsqueeze_249), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %unsqueeze_253), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %unsqueeze_255), kwargs = {})
#   %relu_31 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_72,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 71680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1120)
    x0 = (xindex % 16)
    x2 = xindex // 17920
    x3 = xindex
    tmp45 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 17152*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1120, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 72, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-1024) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 48, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr0 + (16384 + x0 + 16*((-1024) + x1) + 17152*x2), tmp26 & xmask, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 72, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (16384 + x0 + 16*((-48) + ((-1024) + x1)) + 16768*x2), tmp31 & xmask, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 96, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (16384 + x0 + 16*((-72) + ((-1024) + x1)) + 16768*x2), tmp39 & xmask, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, xmask)
    tl.store(out_ptr1 + (x3), tmp61, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rk/crkbljowexz2kd7x72gakf34oq3gkiydb5s7a7ds4ij47jibq26j.py
# Topologically Sorted Source Nodes: [dense_9], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_9 => cat_18
# Graph fragment:
#   %cat_18 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_16, %slice_102], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 120)
    x0 = (xindex % 16)
    x2 = xindex // 1920
    x3 = (xindex % 1920)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 72, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 48, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (16384 + x0 + 16*(x1) + 17152*x2), tmp16 & xmask, other=0.0)
    tmp18 = tmp11 >= tmp14
    tmp19 = tl.full([1], 72, tl.int64)
    tmp20 = tmp11 < tmp19
    tmp21 = tmp18 & tmp10
    tmp22 = tl.load(in_ptr1 + (16384 + x0 + 16*((-48) + (x1)) + 16768*x2), tmp21 & xmask, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tmp5 >= tmp8
    tmp27 = tl.full([1], 96, tl.int64)
    tmp28 = tmp5 < tmp27
    tmp29 = tmp26 & tmp4
    tmp30 = tl.load(in_ptr2 + (16384 + x0 + 16*((-72) + (x1)) + 16768*x2), tmp29 & xmask, other=0.0)
    tmp31 = tl.where(tmp9, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 120, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr3 + (16384 + x0 + 16*((-96) + x1) + 16768*x2), tmp34 & xmask, other=0.0)
    tmp38 = tl.where(tmp4, tmp33, tmp37)
    tl.store(out_ptr0 + (x3 + 18304*x2), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yn/cynick47gmp23vzmdaqxusems7z7qrtzrumfuuyplyw2gmkstl2s.py
# Topologically Sorted Source Nodes: [resid_7, resid_8, resid_9], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_7 => add_63
#   resid_8 => add_70
#   resid_9 => add_77
# Graph fragment:
#   %add_63 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_74, %slice_82), kwargs = {})
#   %add_70 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_63, %slice_90), kwargs = {})
#   %add_77 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_70, %slice_98), kwargs = {})
triton_poi_fused_add_21 = async_compile.triton('triton_poi_fused_add_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16384)
    x1 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x0 + 17152*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 16768*x1), None)
    tmp3 = tl.load(in_ptr2 + (x0 + 16768*x1), None)
    tmp5 = tl.load(in_ptr3 + (x0 + 16768*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0 + 18304*x1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/af/cafvc32fxc7ldatbya5ti6elnaxoujjhjmvbmbbbgzhozq6y322p.py
# Topologically Sorted Source Nodes: [batch_norm_34, relu_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_34 => add_79, mul_103, mul_104, sub_34
#   relu_34 => relu_34
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_19, %unsqueeze_273), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %unsqueeze_277), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_279), kwargs = {})
#   %relu_34 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_79,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 1144)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tj/ctjuexj43spcfzjxywbgysdb273qv7ftduaurcc7dycvckcgb5wh.py
# Topologically Sorted Source Nodes: [x_in_43, batch_norm_37, relu_37], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_37 => add_86, mul_112, mul_113, sub_37
#   relu_37 => relu_37
#   x_in_43 => cat_21
# Graph fragment:
#   %cat_21 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_84, %cat_20], 1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_21, %unsqueeze_297), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %relu_37 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_86,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 74752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1168)
    x0 = (xindex % 16)
    x2 = xindex // 18688
    x3 = xindex
    tmp29 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 18304*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1168, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 120, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr2 + (x0 + 16*((-1024) + x1) + 18304*x2), tmp18 & xmask, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 144, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (16384 + x0 + 16*((-120) + ((-1024) + x1)) + 16768*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mh/cmh72jz7rkqwvxllcyfd26p5ndt276n6kkt72e444a6pi5k5zw7k.py
# Topologically Sorted Source Nodes: [x_in_47, batch_norm_40, relu_40], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_40 => add_93, mul_121, mul_122, sub_40
#   relu_40 => relu_40
#   x_in_47 => cat_23
# Graph fragment:
#   %cat_23 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_91, %cat_22], 1), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_23, %unsqueeze_321), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, %unsqueeze_325), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_122, %unsqueeze_327), kwargs = {})
#   %relu_40 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_93,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1192)
    x0 = (xindex % 16)
    x2 = xindex // 19072
    x3 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 18304*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1192, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 144, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-1024) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 120, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr3 + (x0 + 16*((-1024) + x1) + 18304*x2), tmp26 & xmask, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 144, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (16384 + x0 + 16*((-120) + ((-1024) + x1)) + 16768*x2), tmp31 & xmask, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 168, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (16384 + x0 + 16*((-144) + ((-1024) + x1)) + 16768*x2), tmp39 & xmask, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, xmask)
    tl.store(out_ptr1 + (x3), tmp61, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/te/cte2mjbeclfnyh5n2dtwnipjqavjhrrqgqac3znvh7rfz75ezfaz.py
# Topologically Sorted Source Nodes: [dense_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_12 => cat_24
# Graph fragment:
#   %cat_24 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_22, %slice_126], 1), kwargs = {})
triton_poi_fused_cat_25 = async_compile.triton('triton_poi_fused_cat_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 192)
    x0 = (xindex % 16)
    x2 = xindex // 3072
    x3 = (xindex % 3072)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 168, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 144, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 120, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (x0 + 16*(x1) + 18304*x2), tmp16, other=0.0)
    tmp18 = tmp11 >= tmp14
    tmp19 = tl.full([1], 144, tl.int64)
    tmp20 = tmp11 < tmp19
    tmp21 = tmp18 & tmp10
    tmp22 = tl.load(in_ptr1 + (16384 + x0 + 16*((-120) + (x1)) + 16768*x2), tmp21, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tmp5 >= tmp8
    tmp27 = tl.full([1], 168, tl.int64)
    tmp28 = tmp5 < tmp27
    tmp29 = tmp26 & tmp4
    tmp30 = tl.load(in_ptr2 + (16384 + x0 + 16*((-144) + (x1)) + 16768*x2), tmp29, other=0.0)
    tmp31 = tl.where(tmp9, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 192, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr3 + (16384 + x0 + 16*((-168) + x1) + 16768*x2), tmp34, other=0.0)
    tmp38 = tl.where(tmp4, tmp33, tmp37)
    tl.store(out_ptr0 + (x3 + 19456*x2), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/jr/cjrgbbnhglgk5yjzg5s57vka2zrjdwk4lpfyalmtbxh3xtythvg2.py
# Topologically Sorted Source Nodes: [resid_10, resid_11, resid_12], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_10 => add_84
#   resid_11 => add_91
#   resid_12 => add_98
# Graph fragment:
#   %add_84 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_77, %slice_106), kwargs = {})
#   %add_91 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_84, %slice_114), kwargs = {})
#   %add_98 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %slice_122), kwargs = {})
triton_poi_fused_add_26 = async_compile.triton('triton_poi_fused_add_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16384)
    x1 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x0 + 18304*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 16768*x1), None)
    tmp3 = tl.load(in_ptr2 + (x0 + 16768*x1), None)
    tmp5 = tl.load(in_ptr3 + (x0 + 16768*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0 + 19456*x1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/3l/c3ltufsgf2pgaifaux6u5mdppk7rdem4kqabwu5xupjbc6cpxlq7.py
# Topologically Sorted Source Nodes: [batch_norm_43, relu_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_43 => add_100, mul_130, mul_131, sub_43
#   relu_43 => relu_43
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_25, %unsqueeze_345), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_43 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_100,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 77824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 1216)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ek/cekqystr7jmaeqnk5cqylrck5vlxvvjcla63oujv6j3xakhwwapf.py
# Topologically Sorted Source Nodes: [x_in_55, batch_norm_46, relu_46], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_46 => add_107, mul_139, mul_140, sub_46
#   relu_46 => relu_46
#   x_in_55 => cat_27
# Graph fragment:
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_105, %cat_26], 1), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_27, %unsqueeze_369), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %unsqueeze_373), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_140, %unsqueeze_375), kwargs = {})
#   %relu_46 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 79360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1240)
    x0 = (xindex % 16)
    x2 = xindex // 19840
    x3 = xindex
    tmp29 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 19456*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1240, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 192, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr2 + (x0 + 16*((-1024) + x1) + 19456*x2), tmp18 & xmask, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 216, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (16384 + x0 + 16*((-192) + ((-1024) + x1)) + 16768*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pv/cpvhbe63h7o6gtuulnhd4ruaylqicarvbdgah53rudk4lkwliyc7.py
# Topologically Sorted Source Nodes: [x_in_59, batch_norm_49, relu_49], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_49 => add_114, mul_148, mul_149, sub_49
#   relu_49 => relu_49
#   x_in_59 => cat_29
# Graph fragment:
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_112, %cat_28], 1), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_29, %unsqueeze_393), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_397), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_399), kwargs = {})
#   %relu_49 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_114,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1264)
    x0 = (xindex % 16)
    x2 = xindex // 20224
    x3 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 19456*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1264, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 216, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-1024) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 192, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr3 + (x0 + 16*((-1024) + x1) + 19456*x2), tmp26 & xmask, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 216, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (16384 + x0 + 16*((-192) + ((-1024) + x1)) + 16768*x2), tmp31 & xmask, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 240, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (16384 + x0 + 16*((-216) + ((-1024) + x1)) + 16768*x2), tmp39 & xmask, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, xmask)
    tl.store(out_ptr1 + (x3), tmp61, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gd/cgdnjxy65iqfoylrsahoy4gioqronjcmmn22hnhsk5kwzb5hsalm.py
# Topologically Sorted Source Nodes: [dense_15], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_15 => cat_30
# Graph fragment:
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_28, %slice_150], 1), kwargs = {})
triton_poi_fused_cat_30 = async_compile.triton('triton_poi_fused_cat_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 264)
    x0 = (xindex % 16)
    x2 = xindex // 4224
    x3 = (xindex % 4224)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 216, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 192, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (x0 + 16*(x1) + 19456*x2), tmp16 & xmask, other=0.0)
    tmp18 = tmp11 >= tmp14
    tmp19 = tl.full([1], 216, tl.int64)
    tmp20 = tmp11 < tmp19
    tmp21 = tmp18 & tmp10
    tmp22 = tl.load(in_ptr1 + (16384 + x0 + 16*((-192) + (x1)) + 16768*x2), tmp21 & xmask, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tmp5 >= tmp8
    tmp27 = tl.full([1], 240, tl.int64)
    tmp28 = tmp5 < tmp27
    tmp29 = tmp26 & tmp4
    tmp30 = tl.load(in_ptr2 + (16384 + x0 + 16*((-216) + (x1)) + 16768*x2), tmp29 & xmask, other=0.0)
    tmp31 = tl.where(tmp9, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 264, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr3 + (16384 + x0 + 16*((-240) + x1) + 16768*x2), tmp34 & xmask, other=0.0)
    tmp38 = tl.where(tmp4, tmp33, tmp37)
    tl.store(out_ptr0 + (x3 + 20608*x2), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci4b52yyccek3y6liyoqantahdydvdnft7qpmrpfaz7kb3fz6rus.py
# Topologically Sorted Source Nodes: [resid_13, resid_14, resid_15], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_13 => add_105
#   resid_14 => add_112
#   resid_15 => add_119
# Graph fragment:
#   %add_105 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_98, %slice_130), kwargs = {})
#   %add_112 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_105, %slice_138), kwargs = {})
#   %add_119 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_112, %slice_146), kwargs = {})
triton_poi_fused_add_31 = async_compile.triton('triton_poi_fused_add_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16384)
    x1 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x0 + 19456*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 16768*x1), None)
    tmp3 = tl.load(in_ptr2 + (x0 + 16768*x1), None)
    tmp5 = tl.load(in_ptr3 + (x0 + 16768*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0 + 20608*x1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/la/claz5mt6uht6j5vjtd37ecehgwkwry3tihqbmozo6maawgumvmop.py
# Topologically Sorted Source Nodes: [batch_norm_52, relu_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_52 => add_121, mul_157, mul_158, sub_52
#   relu_52 => relu_52
# Graph fragment:
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_31, %unsqueeze_417), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %unsqueeze_421), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %unsqueeze_423), kwargs = {})
#   %relu_52 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_121,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 82432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 1288)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ze/czexam7kjqhnmn33urlklmto2rdexpoecokjlndd7avtlwdadro7.py
# Topologically Sorted Source Nodes: [x_in_67, batch_norm_55, relu_55], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_55 => add_128, mul_166, mul_167, sub_55
#   relu_55 => relu_55
#   x_in_67 => cat_33
# Graph fragment:
#   %cat_33 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_126, %cat_32], 1), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_33, %unsqueeze_441), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_445), kwargs = {})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_447), kwargs = {})
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_128,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 83968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1312)
    x0 = (xindex % 16)
    x2 = xindex // 20992
    x3 = xindex
    tmp29 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 20608*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1312, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 264, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr2 + (x0 + 16*((-1024) + x1) + 20608*x2), tmp18 & xmask, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 288, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (16384 + x0 + 16*((-264) + ((-1024) + x1)) + 16768*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jw/cjwiao6qgq3kakg6x52dsiz5fq6wvf6wllg6jwkmgvbfkcscnsxn.py
# Topologically Sorted Source Nodes: [x_in_71, batch_norm_58, relu_58], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_58 => add_135, mul_175, mul_176, sub_58
#   relu_58 => relu_58
#   x_in_71 => cat_35
# Graph fragment:
#   %cat_35 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_133, %cat_34], 1), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_35, %unsqueeze_465), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_467), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_469), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_471), kwargs = {})
#   %relu_58 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_135,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 85504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1336)
    x0 = (xindex % 16)
    x2 = xindex // 21376
    x3 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 20608*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1336, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 288, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-1024) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 264, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr3 + (x0 + 16*((-1024) + x1) + 20608*x2), tmp26 & xmask, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 288, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (16384 + x0 + 16*((-264) + ((-1024) + x1)) + 16768*x2), tmp31 & xmask, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 312, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (16384 + x0 + 16*((-288) + ((-1024) + x1)) + 16768*x2), tmp39 & xmask, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, xmask)
    tl.store(out_ptr1 + (x3), tmp61, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ji/cjiuqbs6ujkh74lmbbh3m3j4il2f63fdb4qw3uddnq33hlzxcdcs.py
# Topologically Sorted Source Nodes: [dense_18], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_18 => cat_36
# Graph fragment:
#   %cat_36 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_34, %slice_174], 1), kwargs = {})
triton_poi_fused_cat_35 = async_compile.triton('triton_poi_fused_cat_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 336)
    x0 = (xindex % 16)
    x2 = xindex // 5376
    x3 = (xindex % 5376)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 312, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 288, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 264, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (x0 + 16*(x1) + 20608*x2), tmp16 & xmask, other=0.0)
    tmp18 = tmp11 >= tmp14
    tmp19 = tl.full([1], 288, tl.int64)
    tmp20 = tmp11 < tmp19
    tmp21 = tmp18 & tmp10
    tmp22 = tl.load(in_ptr1 + (16384 + x0 + 16*((-264) + (x1)) + 16768*x2), tmp21 & xmask, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tmp5 >= tmp8
    tmp27 = tl.full([1], 312, tl.int64)
    tmp28 = tmp5 < tmp27
    tmp29 = tmp26 & tmp4
    tmp30 = tl.load(in_ptr2 + (16384 + x0 + 16*((-288) + (x1)) + 16768*x2), tmp29 & xmask, other=0.0)
    tmp31 = tl.where(tmp9, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 336, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr3 + (16384 + x0 + 16*((-312) + x1) + 16768*x2), tmp34 & xmask, other=0.0)
    tmp38 = tl.where(tmp4, tmp33, tmp37)
    tl.store(out_ptr0 + (x3 + 21760*x2), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3g/c3gcysvq5t2pjc2lzsdhdysle45itxl2wdcqlitvkmerrx6tghcb.py
# Topologically Sorted Source Nodes: [resid_16, resid_17, resid_18], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_16 => add_126
#   resid_17 => add_133
#   resid_18 => add_140
# Graph fragment:
#   %add_126 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_119, %slice_154), kwargs = {})
#   %add_133 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_126, %slice_162), kwargs = {})
#   %add_140 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %slice_170), kwargs = {})
triton_poi_fused_add_36 = async_compile.triton('triton_poi_fused_add_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16384)
    x1 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x0 + 20608*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 16768*x1), None)
    tmp3 = tl.load(in_ptr2 + (x0 + 16768*x1), None)
    tmp5 = tl.load(in_ptr3 + (x0 + 16768*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0 + 21760*x1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/a4/ca46gkedunhlv6sfoleclkrllaxia3t3ng3425y44pctqdzvsytr.py
# Topologically Sorted Source Nodes: [batch_norm_61, relu_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_61 => add_142, mul_184, mul_185, sub_61
#   relu_61 => relu_61
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_37, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_142 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_142,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 1360)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r7/cr7tykh4fjqk5hatbhtqtmyukynff546lz4dqvhctzk5b2hq4fe2.py
# Topologically Sorted Source Nodes: [x_in_79, batch_norm_64, relu_64], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_64 => add_149, mul_193, mul_194, sub_64
#   relu_64 => relu_64
#   x_in_79 => cat_39
# Graph fragment:
#   %cat_39 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_147, %cat_38], 1), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_39, %unsqueeze_513), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_193, %unsqueeze_517), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %unsqueeze_519), kwargs = {})
#   %relu_64 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_149,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 88576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1384)
    x0 = (xindex % 16)
    x2 = xindex // 22144
    x3 = xindex
    tmp29 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 21760*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1384, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 336, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr2 + (x0 + 16*((-1024) + x1) + 21760*x2), tmp18 & xmask, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 360, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (16384 + x0 + 16*((-336) + ((-1024) + x1)) + 16768*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hv/chvv3hqpd46futh72e4fsgep5fezx4xxi5orwk7xpw7fjgd3n5yv.py
# Topologically Sorted Source Nodes: [x_in_83, batch_norm_67, relu_67], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_67 => add_156, mul_202, mul_203, sub_67
#   relu_67 => relu_67
#   x_in_83 => cat_41
# Graph fragment:
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_154, %cat_40], 1), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_41, %unsqueeze_537), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_541), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_543), kwargs = {})
#   %relu_67 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_156,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 1408)
    x0 = (xindex % 16)
    x2 = xindex // 22528
    x3 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 21760*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 16*(x1) + 16768*x2), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1408, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 360, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-1024) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 336, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr3 + (x0 + 16*((-1024) + x1) + 21760*x2), tmp26, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 360, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (16384 + x0 + 16*((-336) + ((-1024) + x1)) + 16768*x2), tmp31, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 384, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (16384 + x0 + 16*((-360) + ((-1024) + x1)) + 16768*x2), tmp39, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, None)
    tl.store(out_ptr1 + (x3), tmp61, None)
''', device_str='cuda')


# kernel path: inductor_cache/zk/czkep4gdoszmesbot2eoco3pnyb5smjoyo2biylwk3q2u6rrx72q.py
# Topologically Sorted Source Nodes: [dense_21], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_21 => cat_42
# Graph fragment:
#   %cat_42 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_40, %slice_198], 1), kwargs = {})
triton_poi_fused_cat_40 = async_compile.triton('triton_poi_fused_cat_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 408)
    x0 = (xindex % 16)
    x2 = xindex // 6528
    x3 = (xindex % 6528)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 360, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 336, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (x0 + 16*(x1) + 21760*x2), tmp16 & xmask, other=0.0)
    tmp18 = tmp11 >= tmp14
    tmp19 = tl.full([1], 360, tl.int64)
    tmp20 = tmp11 < tmp19
    tmp21 = tmp18 & tmp10
    tmp22 = tl.load(in_ptr1 + (16384 + x0 + 16*((-336) + (x1)) + 16768*x2), tmp21 & xmask, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tmp5 >= tmp8
    tmp27 = tl.full([1], 384, tl.int64)
    tmp28 = tmp5 < tmp27
    tmp29 = tmp26 & tmp4
    tmp30 = tl.load(in_ptr2 + (16384 + x0 + 16*((-360) + (x1)) + 16768*x2), tmp29 & xmask, other=0.0)
    tmp31 = tl.where(tmp9, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 408, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr3 + (16384 + x0 + 16*((-384) + x1) + 16768*x2), tmp34 & xmask, other=0.0)
    tmp38 = tl.where(tmp4, tmp33, tmp37)
    tl.store(out_ptr0 + (x3 + 22912*x2), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/si/csih3mfu7ctqeocydatrlka43irizorjpzn2dlyr62jg34xsnq6n.py
# Topologically Sorted Source Nodes: [resid_19, resid_20, resid_21], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_19 => add_147
#   resid_20 => add_154
#   resid_21 => add_161
# Graph fragment:
#   %add_147 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_140, %slice_178), kwargs = {})
#   %add_154 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_147, %slice_186), kwargs = {})
#   %add_161 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_154, %slice_194), kwargs = {})
triton_poi_fused_add_41 = async_compile.triton('triton_poi_fused_add_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16384)
    x1 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x0 + 21760*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 16768*x1), None)
    tmp3 = tl.load(in_ptr2 + (x0 + 16768*x1), None)
    tmp5 = tl.load(in_ptr3 + (x0 + 16768*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0 + 22912*x1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/sk/cskqvqpkbp6ivlujd5d3m4rmhg6qnd2momzbjgrxgjamgqp4tnbt.py
# Topologically Sorted Source Nodes: [batch_norm_70, relu_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_70 => add_163, mul_211, mul_212, sub_70
#   relu_70 => relu_70
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_43, %unsqueeze_561), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_211, %unsqueeze_565), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_212, %unsqueeze_567), kwargs = {})
#   %relu_70 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_163,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 91648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 1432)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7c/c7czbztxmyrlwaf6xh2gr7ie3mjpygifgiut2jworf5gpwti4qnl.py
# Topologically Sorted Source Nodes: [x_in_91, batch_norm_73, relu_73], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_73 => add_170, mul_220, mul_221, sub_73
#   relu_73 => relu_73
#   x_in_91 => cat_45
# Graph fragment:
#   %cat_45 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_168, %cat_44], 1), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_45, %unsqueeze_585), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_587), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_220, %unsqueeze_589), kwargs = {})
#   %add_170 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_221, %unsqueeze_591), kwargs = {})
#   %relu_73 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_170,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 93184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1456)
    x0 = (xindex % 16)
    x2 = xindex // 23296
    x3 = xindex
    tmp29 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 22912*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1456, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 408, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr2 + (x0 + 16*((-1024) + x1) + 22912*x2), tmp18 & xmask, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 432, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (16384 + x0 + 16*((-408) + ((-1024) + x1)) + 16768*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l5/cl5czlo6mxkzfnq5sai4mj4keid7kn7hb5cvzlotlnvg3opekyx7.py
# Topologically Sorted Source Nodes: [x_in_95, batch_norm_76, relu_76], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_76 => add_177, mul_229, mul_230, sub_76
#   relu_76 => relu_76
#   x_in_95 => cat_47
# Graph fragment:
#   %cat_47 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_175, %cat_46], 1), kwargs = {})
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_47, %unsqueeze_609), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_611), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_229, %unsqueeze_613), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_230, %unsqueeze_615), kwargs = {})
#   %relu_76 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_177,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1480)
    x0 = (xindex % 16)
    x2 = xindex // 23680
    x3 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 22912*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1480, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 432, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-1024) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 408, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr3 + (x0 + 16*((-1024) + x1) + 22912*x2), tmp26 & xmask, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 432, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (16384 + x0 + 16*((-408) + ((-1024) + x1)) + 16768*x2), tmp31 & xmask, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 456, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (16384 + x0 + 16*((-432) + ((-1024) + x1)) + 16768*x2), tmp39 & xmask, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, xmask)
    tl.store(out_ptr1 + (x3), tmp61, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s2/cs2yiwiarrhknrtzvx6yaxkriba4riniv52k366u7hmavk4ii4fc.py
# Topologically Sorted Source Nodes: [dense_24], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_24 => cat_48
# Graph fragment:
#   %cat_48 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_46, %slice_222], 1), kwargs = {})
triton_poi_fused_cat_45 = async_compile.triton('triton_poi_fused_cat_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 480)
    x0 = (xindex % 16)
    x2 = xindex // 7680
    x3 = (xindex % 7680)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 456, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 432, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 408, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (x0 + 16*(x1) + 22912*x2), tmp16 & xmask, other=0.0)
    tmp18 = tmp11 >= tmp14
    tmp19 = tl.full([1], 432, tl.int64)
    tmp20 = tmp11 < tmp19
    tmp21 = tmp18 & tmp10
    tmp22 = tl.load(in_ptr1 + (16384 + x0 + 16*((-408) + (x1)) + 16768*x2), tmp21 & xmask, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tmp5 >= tmp8
    tmp27 = tl.full([1], 456, tl.int64)
    tmp28 = tmp5 < tmp27
    tmp29 = tmp26 & tmp4
    tmp30 = tl.load(in_ptr2 + (16384 + x0 + 16*((-432) + (x1)) + 16768*x2), tmp29 & xmask, other=0.0)
    tmp31 = tl.where(tmp9, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 480, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr3 + (16384 + x0 + 16*((-456) + x1) + 16768*x2), tmp34 & xmask, other=0.0)
    tmp38 = tl.where(tmp4, tmp33, tmp37)
    tl.store(out_ptr0 + (x3 + 24064*x2), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mg/cmg5pyfud3ampktb2eokzmjvaigdbfjd2kecteomc5vpuinusoad.py
# Topologically Sorted Source Nodes: [resid_22, resid_23, resid_24], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_22 => add_168
#   resid_23 => add_175
#   resid_24 => add_182
# Graph fragment:
#   %add_168 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_161, %slice_202), kwargs = {})
#   %add_175 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_168, %slice_210), kwargs = {})
#   %add_182 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_175, %slice_218), kwargs = {})
triton_poi_fused_add_46 = async_compile.triton('triton_poi_fused_add_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16384)
    x1 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x0 + 22912*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 16768*x1), None)
    tmp3 = tl.load(in_ptr2 + (x0 + 16768*x1), None)
    tmp5 = tl.load(in_ptr3 + (x0 + 16768*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0 + 24064*x1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/zf/czfz7k2pbp3bzvr7qo5g62pqdk5k2jcd26velt4h4s7r6uqahf2u.py
# Topologically Sorted Source Nodes: [batch_norm_79, relu_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_79 => add_184, mul_238, mul_239, sub_79
#   relu_79 => relu_79
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_49, %unsqueeze_633), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_637), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_639), kwargs = {})
#   %relu_79 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_184,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 1504)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ps/cpso5vkuqnqrnujz6amwrajyg2op7hnd7nz2qum7wubbbjbglp5r.py
# Topologically Sorted Source Nodes: [x_in_103, batch_norm_82, relu_82], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_82 => add_191, mul_247, mul_248, sub_82
#   relu_82 => relu_82
#   x_in_103 => cat_51
# Graph fragment:
#   %cat_51 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_189, %cat_50], 1), kwargs = {})
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_51, %unsqueeze_657), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %unsqueeze_659), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_247, %unsqueeze_661), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_248, %unsqueeze_663), kwargs = {})
#   %relu_82 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_191,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 97792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1528)
    x0 = (xindex % 16)
    x2 = xindex // 24448
    x3 = xindex
    tmp29 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 24064*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1528, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 480, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr2 + (x0 + 16*((-1024) + x1) + 24064*x2), tmp18 & xmask, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 504, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (16384 + x0 + 16*((-480) + ((-1024) + x1)) + 16768*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p3/cp3oz5bdfnkm4qy7p3ggq7cu6bqcxzko67y2c2hueyeen3fxoleh.py
# Topologically Sorted Source Nodes: [x_in_107, batch_norm_85, relu_85, batch_norm_86, relu_86], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_85 => add_198, mul_256, mul_257, sub_85
#   batch_norm_86 => add_200, mul_259, mul_260, sub_86
#   relu_85 => relu_85
#   relu_86 => relu_86
#   x_in_107 => cat_53
# Graph fragment:
#   %cat_53 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_196, %cat_52], 1), kwargs = {})
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_53, %unsqueeze_681), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_685), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_687), kwargs = {})
#   %relu_85 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_198,), kwargs = {})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_53, %unsqueeze_689), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %unsqueeze_691), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_259, %unsqueeze_693), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_260, %unsqueeze_695), kwargs = {})
#   %relu_86 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_200,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_49', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 99328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1552)
    x0 = (xindex % 16)
    x2 = xindex // 24832
    x3 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 24064*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1552, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 504, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-1024) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 480, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr3 + (x0 + 16*((-1024) + x1) + 24064*x2), tmp26 & xmask, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 504, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (16384 + x0 + 16*((-480) + ((-1024) + x1)) + 16768*x2), tmp31 & xmask, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 528, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (16384 + x0 + 16*((-504) + ((-1024) + x1)) + 16768*x2), tmp39 & xmask, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tmp63 = tmp44 - tmp62
    tmp65 = tmp64 + tmp48
    tmp66 = libdevice.sqrt(tmp65)
    tmp67 = tmp51 / tmp66
    tmp68 = tmp67 * tmp53
    tmp69 = tmp63 * tmp68
    tmp71 = tmp69 * tmp70
    tmp73 = tmp71 + tmp72
    tmp74 = triton_helpers.maximum(tmp60, tmp73)
    tl.store(out_ptr0 + (x3), tmp44, xmask)
    tl.store(out_ptr1 + (x3), tmp61, xmask)
    tl.store(out_ptr2 + (x3), tmp74, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rx/crx3wmmqiyfckao6z3xsg2imawpjwq53jepfzbibvg5impfc7ekw.py
# Topologically Sorted Source Nodes: [batch_norm_87, relu_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_87 => add_202, mul_262, mul_263, sub_87
#   relu_87 => relu_87
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_86, %unsqueeze_697), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_701), kwargs = {})
#   %add_202 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_703), kwargs = {})
#   %relu_87 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_202,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 768)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/6y/c6ycwjc36jfvkrmfzkwfnflscn6q2wgt2keqv2y4bv7j2bnivj2i.py
# Topologically Sorted Source Nodes: [batch_norm_88, relu_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_88 => add_204, mul_265, mul_266, sub_88
#   relu_88 => relu_88
# Graph fragment:
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_705), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_709), kwargs = {})
#   %add_204 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_711), kwargs = {})
#   %relu_88 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_204,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_51', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 768)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/k7/ck7ooo3w7cobtadxrvm7poasbxhc65aihp4cg25em47pvntpoox5.py
# Topologically Sorted Source Nodes: [x_in_111, batch_norm_89, relu_89], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_89 => add_207, mul_268, mul_269, sub_89
#   relu_89 => relu_89
#   x_in_111 => cat_55
# Graph fragment:
#   %cat_55 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_205, %cat_54], 1), kwargs = {})
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_55, %unsqueeze_713), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_268, %unsqueeze_717), kwargs = {})
#   %add_207 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_269, %unsqueeze_719), kwargs = {})
#   %relu_89 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_207,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 2432)
    x0 = (xindex % 4)
    x2 = xindex // 9728
    x3 = xindex
    tmp29 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 9216*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 4*(x1) + 8704*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 2432, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-2048) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 256, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr0 + (8192 + x0 + 4*((-2048) + x1) + 9216*x2), tmp18 & xmask, other=0.0)
    tmp20 = tmp13 >= tmp16
    tmp21 = tl.full([1], 384, tl.int64)
    tmp22 = tmp13 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr1 + (8192 + x0 + 4*((-256) + ((-2048) + x1)) + 8704*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.where(tmp17, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ta/cta44t2hfruc2nrf35zwtabyl4expjhaq7lkdxbewrz2yzg4ntc7.py
# Topologically Sorted Source Nodes: [x_in_115, batch_norm_92, relu_92], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_92 => add_214, mul_277, mul_278, sub_92
#   relu_92 => relu_92
#   x_in_115 => cat_57
# Graph fragment:
#   %cat_57 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_212, %cat_56], 1), kwargs = {})
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_57, %unsqueeze_737), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_277, %unsqueeze_741), kwargs = {})
#   %add_214 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_278, %unsqueeze_743), kwargs = {})
#   %relu_92 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_214,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 2560)
    x0 = (xindex % 4)
    x2 = xindex // 10240
    x3 = xindex
    tmp45 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 9216*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 4*(x1) + 8704*x2), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 4*(x1) + 8704*x2), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 2560, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-2048) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 384, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-2048) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 256, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr0 + (8192 + x0 + 4*((-2048) + x1) + 9216*x2), tmp26, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 384, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (8192 + x0 + 4*((-256) + ((-2048) + x1)) + 8704*x2), tmp31, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 512, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (8192 + x0 + 4*((-384) + ((-2048) + x1)) + 8704*x2), tmp39, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, None)
    tl.store(out_ptr1 + (x3), tmp61, None)
''', device_str='cuda')


# kernel path: inductor_cache/fb/cfbnbblp6y4k4a6pxtsqzk64knecf26fgiackji6qsgtlbeh4fzh.py
# Topologically Sorted Source Nodes: [dense_29], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_29 => cat_58
# Graph fragment:
#   %cat_58 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_56, %slice_270], 1), kwargs = {})
triton_poi_fused_cat_54 = async_compile.triton('triton_poi_fused_cat_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 640)
    x0 = (xindex % 4)
    x2 = xindex // 2560
    x3 = (xindex % 2560)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 384, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 256, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (8192 + x0 + 4*(x1) + 9216*x2), tmp16 & xmask, other=0.0)
    tmp18 = tmp11 >= tmp14
    tmp19 = tl.full([1], 384, tl.int64)
    tmp20 = tmp11 < tmp19
    tmp21 = tmp18 & tmp10
    tmp22 = tl.load(in_ptr1 + (8192 + x0 + 4*((-256) + (x1)) + 8704*x2), tmp21 & xmask, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tmp5 >= tmp8
    tmp27 = tl.full([1], 512, tl.int64)
    tmp28 = tmp5 < tmp27
    tmp29 = tmp26 & tmp4
    tmp30 = tl.load(in_ptr2 + (8192 + x0 + 4*((-384) + (x1)) + 8704*x2), tmp29 & xmask, other=0.0)
    tmp31 = tl.where(tmp9, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 640, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr3 + (8192 + x0 + 4*((-512) + x1) + 8704*x2), tmp34 & xmask, other=0.0)
    tmp38 = tl.where(tmp4, tmp33, tmp37)
    tl.store(out_ptr0 + (x3 + 10752*x2), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ka/ckakyiqpbantbfpwqglrxufvon5cjal2i4irpt7ltgartlmogeps.py
# Topologically Sorted Source Nodes: [resid_27, resid_28, resid_29], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_27 => add_205
#   resid_28 => add_212
#   resid_29 => add_219
# Graph fragment:
#   %add_205 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_242, %slice_250), kwargs = {})
#   %add_212 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_205, %slice_258), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_212, %slice_266), kwargs = {})
triton_poi_fused_add_55 = async_compile.triton('triton_poi_fused_add_55', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8192)
    x1 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (x0 + 9216*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 8704*x1), None)
    tmp3 = tl.load(in_ptr2 + (x0 + 8704*x1), None)
    tmp5 = tl.load(in_ptr3 + (x0 + 8704*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0 + 10752*x1), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/fw/cfw5f7az5465f2724sq7hb5jpngvnqfsb4d22oafr4nrszgfbpww.py
# Topologically Sorted Source Nodes: [batch_norm_95, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_95 => add_221, mul_286, mul_287, sub_95
#   input_1 => relu_95
# Graph fragment:
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_59, %unsqueeze_761), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_286, %unsqueeze_765), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_287, %unsqueeze_767), kwargs = {})
#   %relu_95 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_221,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_56', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 2688)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vw/cvw5mebnrmew4zccxovt3zgqk3dnspqgbdydnmiyawa7irr4jl2d.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_5 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_95, [2, 2], [], [0, 0], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_57 = async_compile.triton('triton_poi_fused_avg_pool2d_57', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_57(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/56/c56msjw4v2kwbpve2b34jifooamzbkciwisknt5xvqmleevc5zz2.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out => convolution_95
# Graph fragment:
#   %convolution_95 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d, %primals_481, %primals_482, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_58 = async_compile.triton('triton_poi_fused_convolution_58', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_58(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1000)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (288, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (96, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_17, (96, ), (1, ))
    assert_size_stride(primals_18, (96, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_20, (96, ), (1, ))
    assert_size_stride(primals_21, (96, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_22, (96, ), (1, ))
    assert_size_stride(primals_23, (96, ), (1, ))
    assert_size_stride(primals_24, (96, ), (1, ))
    assert_size_stride(primals_25, (96, ), (1, ))
    assert_size_stride(primals_26, (272, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_27, (304, ), (1, ))
    assert_size_stride(primals_28, (304, ), (1, ))
    assert_size_stride(primals_29, (304, ), (1, ))
    assert_size_stride(primals_30, (304, ), (1, ))
    assert_size_stride(primals_31, (96, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_32, (96, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_36, (96, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_37, (96, ), (1, ))
    assert_size_stride(primals_38, (96, ), (1, ))
    assert_size_stride(primals_39, (96, ), (1, ))
    assert_size_stride(primals_40, (96, ), (1, ))
    assert_size_stride(primals_41, (272, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_42, (320, ), (1, ))
    assert_size_stride(primals_43, (320, ), (1, ))
    assert_size_stride(primals_44, (320, ), (1, ))
    assert_size_stride(primals_45, (320, ), (1, ))
    assert_size_stride(primals_46, (96, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_47, (96, ), (1, ))
    assert_size_stride(primals_48, (96, ), (1, ))
    assert_size_stride(primals_49, (96, ), (1, ))
    assert_size_stride(primals_50, (96, ), (1, ))
    assert_size_stride(primals_51, (96, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_52, (96, ), (1, ))
    assert_size_stride(primals_53, (96, ), (1, ))
    assert_size_stride(primals_54, (96, ), (1, ))
    assert_size_stride(primals_55, (96, ), (1, ))
    assert_size_stride(primals_56, (272, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_57, (336, ), (1, ))
    assert_size_stride(primals_58, (336, ), (1, ))
    assert_size_stride(primals_59, (336, ), (1, ))
    assert_size_stride(primals_60, (336, ), (1, ))
    assert_size_stride(primals_61, (576, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_62, (336, ), (1, ))
    assert_size_stride(primals_63, (336, ), (1, ))
    assert_size_stride(primals_64, (336, ), (1, ))
    assert_size_stride(primals_65, (336, ), (1, ))
    assert_size_stride(primals_66, (192, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_67, (192, ), (1, ))
    assert_size_stride(primals_68, (192, ), (1, ))
    assert_size_stride(primals_69, (192, ), (1, ))
    assert_size_stride(primals_70, (192, ), (1, ))
    assert_size_stride(primals_71, (192, 6, 3, 3), (54, 9, 3, 1))
    assert_size_stride(primals_72, (192, ), (1, ))
    assert_size_stride(primals_73, (192, ), (1, ))
    assert_size_stride(primals_74, (192, ), (1, ))
    assert_size_stride(primals_75, (192, ), (1, ))
    assert_size_stride(primals_76, (544, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_77, (608, ), (1, ))
    assert_size_stride(primals_78, (608, ), (1, ))
    assert_size_stride(primals_79, (608, ), (1, ))
    assert_size_stride(primals_80, (608, ), (1, ))
    assert_size_stride(primals_81, (192, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(primals_82, (192, ), (1, ))
    assert_size_stride(primals_83, (192, ), (1, ))
    assert_size_stride(primals_84, (192, ), (1, ))
    assert_size_stride(primals_85, (192, ), (1, ))
    assert_size_stride(primals_86, (192, 6, 3, 3), (54, 9, 3, 1))
    assert_size_stride(primals_87, (192, ), (1, ))
    assert_size_stride(primals_88, (192, ), (1, ))
    assert_size_stride(primals_89, (192, ), (1, ))
    assert_size_stride(primals_90, (192, ), (1, ))
    assert_size_stride(primals_91, (544, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_92, (640, ), (1, ))
    assert_size_stride(primals_93, (640, ), (1, ))
    assert_size_stride(primals_94, (640, ), (1, ))
    assert_size_stride(primals_95, (640, ), (1, ))
    assert_size_stride(primals_96, (192, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_97, (192, ), (1, ))
    assert_size_stride(primals_98, (192, ), (1, ))
    assert_size_stride(primals_99, (192, ), (1, ))
    assert_size_stride(primals_100, (192, ), (1, ))
    assert_size_stride(primals_101, (192, 6, 3, 3), (54, 9, 3, 1))
    assert_size_stride(primals_102, (192, ), (1, ))
    assert_size_stride(primals_103, (192, ), (1, ))
    assert_size_stride(primals_104, (192, ), (1, ))
    assert_size_stride(primals_105, (192, ), (1, ))
    assert_size_stride(primals_106, (544, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_107, (672, ), (1, ))
    assert_size_stride(primals_108, (672, ), (1, ))
    assert_size_stride(primals_109, (672, ), (1, ))
    assert_size_stride(primals_110, (672, ), (1, ))
    assert_size_stride(primals_111, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_112, (192, ), (1, ))
    assert_size_stride(primals_113, (192, ), (1, ))
    assert_size_stride(primals_114, (192, ), (1, ))
    assert_size_stride(primals_115, (192, ), (1, ))
    assert_size_stride(primals_116, (192, 6, 3, 3), (54, 9, 3, 1))
    assert_size_stride(primals_117, (192, ), (1, ))
    assert_size_stride(primals_118, (192, ), (1, ))
    assert_size_stride(primals_119, (192, ), (1, ))
    assert_size_stride(primals_120, (192, ), (1, ))
    assert_size_stride(primals_121, (544, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_122, (704, ), (1, ))
    assert_size_stride(primals_123, (704, ), (1, ))
    assert_size_stride(primals_124, (704, ), (1, ))
    assert_size_stride(primals_125, (704, ), (1, ))
    assert_size_stride(primals_126, (1072, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_127, (704, ), (1, ))
    assert_size_stride(primals_128, (704, ), (1, ))
    assert_size_stride(primals_129, (704, ), (1, ))
    assert_size_stride(primals_130, (704, ), (1, ))
    assert_size_stride(primals_131, (384, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_132, (384, ), (1, ))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_137, (384, ), (1, ))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (384, ), (1, ))
    assert_size_stride(primals_141, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_142, (1096, ), (1, ))
    assert_size_stride(primals_143, (1096, ), (1, ))
    assert_size_stride(primals_144, (1096, ), (1, ))
    assert_size_stride(primals_145, (1096, ), (1, ))
    assert_size_stride(primals_146, (384, 1096, 1, 1), (1096, 1, 1, 1))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_152, (384, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (384, ), (1, ))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_157, (1120, ), (1, ))
    assert_size_stride(primals_158, (1120, ), (1, ))
    assert_size_stride(primals_159, (1120, ), (1, ))
    assert_size_stride(primals_160, (1120, ), (1, ))
    assert_size_stride(primals_161, (384, 1120, 1, 1), (1120, 1, 1, 1))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_164, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_172, (1144, ), (1, ))
    assert_size_stride(primals_173, (1144, ), (1, ))
    assert_size_stride(primals_174, (1144, ), (1, ))
    assert_size_stride(primals_175, (1144, ), (1, ))
    assert_size_stride(primals_176, (384, 1144, 1, 1), (1144, 1, 1, 1))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (384, ), (1, ))
    assert_size_stride(primals_180, (384, ), (1, ))
    assert_size_stride(primals_181, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_182, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_184, (384, ), (1, ))
    assert_size_stride(primals_185, (384, ), (1, ))
    assert_size_stride(primals_186, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_187, (1168, ), (1, ))
    assert_size_stride(primals_188, (1168, ), (1, ))
    assert_size_stride(primals_189, (1168, ), (1, ))
    assert_size_stride(primals_190, (1168, ), (1, ))
    assert_size_stride(primals_191, (384, 1168, 1, 1), (1168, 1, 1, 1))
    assert_size_stride(primals_192, (384, ), (1, ))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_197, (384, ), (1, ))
    assert_size_stride(primals_198, (384, ), (1, ))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_200, (384, ), (1, ))
    assert_size_stride(primals_201, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_202, (1192, ), (1, ))
    assert_size_stride(primals_203, (1192, ), (1, ))
    assert_size_stride(primals_204, (1192, ), (1, ))
    assert_size_stride(primals_205, (1192, ), (1, ))
    assert_size_stride(primals_206, (384, 1192, 1, 1), (1192, 1, 1, 1))
    assert_size_stride(primals_207, (384, ), (1, ))
    assert_size_stride(primals_208, (384, ), (1, ))
    assert_size_stride(primals_209, (384, ), (1, ))
    assert_size_stride(primals_210, (384, ), (1, ))
    assert_size_stride(primals_211, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_212, (384, ), (1, ))
    assert_size_stride(primals_213, (384, ), (1, ))
    assert_size_stride(primals_214, (384, ), (1, ))
    assert_size_stride(primals_215, (384, ), (1, ))
    assert_size_stride(primals_216, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_217, (1216, ), (1, ))
    assert_size_stride(primals_218, (1216, ), (1, ))
    assert_size_stride(primals_219, (1216, ), (1, ))
    assert_size_stride(primals_220, (1216, ), (1, ))
    assert_size_stride(primals_221, (384, 1216, 1, 1), (1216, 1, 1, 1))
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_223, (384, ), (1, ))
    assert_size_stride(primals_224, (384, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_227, (384, ), (1, ))
    assert_size_stride(primals_228, (384, ), (1, ))
    assert_size_stride(primals_229, (384, ), (1, ))
    assert_size_stride(primals_230, (384, ), (1, ))
    assert_size_stride(primals_231, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_232, (1240, ), (1, ))
    assert_size_stride(primals_233, (1240, ), (1, ))
    assert_size_stride(primals_234, (1240, ), (1, ))
    assert_size_stride(primals_235, (1240, ), (1, ))
    assert_size_stride(primals_236, (384, 1240, 1, 1), (1240, 1, 1, 1))
    assert_size_stride(primals_237, (384, ), (1, ))
    assert_size_stride(primals_238, (384, ), (1, ))
    assert_size_stride(primals_239, (384, ), (1, ))
    assert_size_stride(primals_240, (384, ), (1, ))
    assert_size_stride(primals_241, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_242, (384, ), (1, ))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_244, (384, ), (1, ))
    assert_size_stride(primals_245, (384, ), (1, ))
    assert_size_stride(primals_246, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_247, (1264, ), (1, ))
    assert_size_stride(primals_248, (1264, ), (1, ))
    assert_size_stride(primals_249, (1264, ), (1, ))
    assert_size_stride(primals_250, (1264, ), (1, ))
    assert_size_stride(primals_251, (384, 1264, 1, 1), (1264, 1, 1, 1))
    assert_size_stride(primals_252, (384, ), (1, ))
    assert_size_stride(primals_253, (384, ), (1, ))
    assert_size_stride(primals_254, (384, ), (1, ))
    assert_size_stride(primals_255, (384, ), (1, ))
    assert_size_stride(primals_256, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_257, (384, ), (1, ))
    assert_size_stride(primals_258, (384, ), (1, ))
    assert_size_stride(primals_259, (384, ), (1, ))
    assert_size_stride(primals_260, (384, ), (1, ))
    assert_size_stride(primals_261, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_262, (1288, ), (1, ))
    assert_size_stride(primals_263, (1288, ), (1, ))
    assert_size_stride(primals_264, (1288, ), (1, ))
    assert_size_stride(primals_265, (1288, ), (1, ))
    assert_size_stride(primals_266, (384, 1288, 1, 1), (1288, 1, 1, 1))
    assert_size_stride(primals_267, (384, ), (1, ))
    assert_size_stride(primals_268, (384, ), (1, ))
    assert_size_stride(primals_269, (384, ), (1, ))
    assert_size_stride(primals_270, (384, ), (1, ))
    assert_size_stride(primals_271, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_272, (384, ), (1, ))
    assert_size_stride(primals_273, (384, ), (1, ))
    assert_size_stride(primals_274, (384, ), (1, ))
    assert_size_stride(primals_275, (384, ), (1, ))
    assert_size_stride(primals_276, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_277, (1312, ), (1, ))
    assert_size_stride(primals_278, (1312, ), (1, ))
    assert_size_stride(primals_279, (1312, ), (1, ))
    assert_size_stride(primals_280, (1312, ), (1, ))
    assert_size_stride(primals_281, (384, 1312, 1, 1), (1312, 1, 1, 1))
    assert_size_stride(primals_282, (384, ), (1, ))
    assert_size_stride(primals_283, (384, ), (1, ))
    assert_size_stride(primals_284, (384, ), (1, ))
    assert_size_stride(primals_285, (384, ), (1, ))
    assert_size_stride(primals_286, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_287, (384, ), (1, ))
    assert_size_stride(primals_288, (384, ), (1, ))
    assert_size_stride(primals_289, (384, ), (1, ))
    assert_size_stride(primals_290, (384, ), (1, ))
    assert_size_stride(primals_291, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_292, (1336, ), (1, ))
    assert_size_stride(primals_293, (1336, ), (1, ))
    assert_size_stride(primals_294, (1336, ), (1, ))
    assert_size_stride(primals_295, (1336, ), (1, ))
    assert_size_stride(primals_296, (384, 1336, 1, 1), (1336, 1, 1, 1))
    assert_size_stride(primals_297, (384, ), (1, ))
    assert_size_stride(primals_298, (384, ), (1, ))
    assert_size_stride(primals_299, (384, ), (1, ))
    assert_size_stride(primals_300, (384, ), (1, ))
    assert_size_stride(primals_301, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_302, (384, ), (1, ))
    assert_size_stride(primals_303, (384, ), (1, ))
    assert_size_stride(primals_304, (384, ), (1, ))
    assert_size_stride(primals_305, (384, ), (1, ))
    assert_size_stride(primals_306, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_307, (1360, ), (1, ))
    assert_size_stride(primals_308, (1360, ), (1, ))
    assert_size_stride(primals_309, (1360, ), (1, ))
    assert_size_stride(primals_310, (1360, ), (1, ))
    assert_size_stride(primals_311, (384, 1360, 1, 1), (1360, 1, 1, 1))
    assert_size_stride(primals_312, (384, ), (1, ))
    assert_size_stride(primals_313, (384, ), (1, ))
    assert_size_stride(primals_314, (384, ), (1, ))
    assert_size_stride(primals_315, (384, ), (1, ))
    assert_size_stride(primals_316, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_317, (384, ), (1, ))
    assert_size_stride(primals_318, (384, ), (1, ))
    assert_size_stride(primals_319, (384, ), (1, ))
    assert_size_stride(primals_320, (384, ), (1, ))
    assert_size_stride(primals_321, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_322, (1384, ), (1, ))
    assert_size_stride(primals_323, (1384, ), (1, ))
    assert_size_stride(primals_324, (1384, ), (1, ))
    assert_size_stride(primals_325, (1384, ), (1, ))
    assert_size_stride(primals_326, (384, 1384, 1, 1), (1384, 1, 1, 1))
    assert_size_stride(primals_327, (384, ), (1, ))
    assert_size_stride(primals_328, (384, ), (1, ))
    assert_size_stride(primals_329, (384, ), (1, ))
    assert_size_stride(primals_330, (384, ), (1, ))
    assert_size_stride(primals_331, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_332, (384, ), (1, ))
    assert_size_stride(primals_333, (384, ), (1, ))
    assert_size_stride(primals_334, (384, ), (1, ))
    assert_size_stride(primals_335, (384, ), (1, ))
    assert_size_stride(primals_336, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_337, (1408, ), (1, ))
    assert_size_stride(primals_338, (1408, ), (1, ))
    assert_size_stride(primals_339, (1408, ), (1, ))
    assert_size_stride(primals_340, (1408, ), (1, ))
    assert_size_stride(primals_341, (384, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(primals_342, (384, ), (1, ))
    assert_size_stride(primals_343, (384, ), (1, ))
    assert_size_stride(primals_344, (384, ), (1, ))
    assert_size_stride(primals_345, (384, ), (1, ))
    assert_size_stride(primals_346, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_347, (384, ), (1, ))
    assert_size_stride(primals_348, (384, ), (1, ))
    assert_size_stride(primals_349, (384, ), (1, ))
    assert_size_stride(primals_350, (384, ), (1, ))
    assert_size_stride(primals_351, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_352, (1432, ), (1, ))
    assert_size_stride(primals_353, (1432, ), (1, ))
    assert_size_stride(primals_354, (1432, ), (1, ))
    assert_size_stride(primals_355, (1432, ), (1, ))
    assert_size_stride(primals_356, (384, 1432, 1, 1), (1432, 1, 1, 1))
    assert_size_stride(primals_357, (384, ), (1, ))
    assert_size_stride(primals_358, (384, ), (1, ))
    assert_size_stride(primals_359, (384, ), (1, ))
    assert_size_stride(primals_360, (384, ), (1, ))
    assert_size_stride(primals_361, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_362, (384, ), (1, ))
    assert_size_stride(primals_363, (384, ), (1, ))
    assert_size_stride(primals_364, (384, ), (1, ))
    assert_size_stride(primals_365, (384, ), (1, ))
    assert_size_stride(primals_366, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_367, (1456, ), (1, ))
    assert_size_stride(primals_368, (1456, ), (1, ))
    assert_size_stride(primals_369, (1456, ), (1, ))
    assert_size_stride(primals_370, (1456, ), (1, ))
    assert_size_stride(primals_371, (384, 1456, 1, 1), (1456, 1, 1, 1))
    assert_size_stride(primals_372, (384, ), (1, ))
    assert_size_stride(primals_373, (384, ), (1, ))
    assert_size_stride(primals_374, (384, ), (1, ))
    assert_size_stride(primals_375, (384, ), (1, ))
    assert_size_stride(primals_376, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_377, (384, ), (1, ))
    assert_size_stride(primals_378, (384, ), (1, ))
    assert_size_stride(primals_379, (384, ), (1, ))
    assert_size_stride(primals_380, (384, ), (1, ))
    assert_size_stride(primals_381, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_382, (1480, ), (1, ))
    assert_size_stride(primals_383, (1480, ), (1, ))
    assert_size_stride(primals_384, (1480, ), (1, ))
    assert_size_stride(primals_385, (1480, ), (1, ))
    assert_size_stride(primals_386, (384, 1480, 1, 1), (1480, 1, 1, 1))
    assert_size_stride(primals_387, (384, ), (1, ))
    assert_size_stride(primals_388, (384, ), (1, ))
    assert_size_stride(primals_389, (384, ), (1, ))
    assert_size_stride(primals_390, (384, ), (1, ))
    assert_size_stride(primals_391, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_392, (384, ), (1, ))
    assert_size_stride(primals_393, (384, ), (1, ))
    assert_size_stride(primals_394, (384, ), (1, ))
    assert_size_stride(primals_395, (384, ), (1, ))
    assert_size_stride(primals_396, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_397, (1504, ), (1, ))
    assert_size_stride(primals_398, (1504, ), (1, ))
    assert_size_stride(primals_399, (1504, ), (1, ))
    assert_size_stride(primals_400, (1504, ), (1, ))
    assert_size_stride(primals_401, (384, 1504, 1, 1), (1504, 1, 1, 1))
    assert_size_stride(primals_402, (384, ), (1, ))
    assert_size_stride(primals_403, (384, ), (1, ))
    assert_size_stride(primals_404, (384, ), (1, ))
    assert_size_stride(primals_405, (384, ), (1, ))
    assert_size_stride(primals_406, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_407, (384, ), (1, ))
    assert_size_stride(primals_408, (384, ), (1, ))
    assert_size_stride(primals_409, (384, ), (1, ))
    assert_size_stride(primals_410, (384, ), (1, ))
    assert_size_stride(primals_411, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_412, (1528, ), (1, ))
    assert_size_stride(primals_413, (1528, ), (1, ))
    assert_size_stride(primals_414, (1528, ), (1, ))
    assert_size_stride(primals_415, (1528, ), (1, ))
    assert_size_stride(primals_416, (384, 1528, 1, 1), (1528, 1, 1, 1))
    assert_size_stride(primals_417, (384, ), (1, ))
    assert_size_stride(primals_418, (384, ), (1, ))
    assert_size_stride(primals_419, (384, ), (1, ))
    assert_size_stride(primals_420, (384, ), (1, ))
    assert_size_stride(primals_421, (384, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_422, (384, ), (1, ))
    assert_size_stride(primals_423, (384, ), (1, ))
    assert_size_stride(primals_424, (384, ), (1, ))
    assert_size_stride(primals_425, (384, ), (1, ))
    assert_size_stride(primals_426, (1048, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_427, (1552, ), (1, ))
    assert_size_stride(primals_428, (1552, ), (1, ))
    assert_size_stride(primals_429, (1552, ), (1, ))
    assert_size_stride(primals_430, (1552, ), (1, ))
    assert_size_stride(primals_431, (2304, 1552, 1, 1), (1552, 1, 1, 1))
    assert_size_stride(primals_432, (1552, ), (1, ))
    assert_size_stride(primals_433, (1552, ), (1, ))
    assert_size_stride(primals_434, (1552, ), (1, ))
    assert_size_stride(primals_435, (1552, ), (1, ))
    assert_size_stride(primals_436, (768, 1552, 1, 1), (1552, 1, 1, 1))
    assert_size_stride(primals_437, (768, ), (1, ))
    assert_size_stride(primals_438, (768, ), (1, ))
    assert_size_stride(primals_439, (768, ), (1, ))
    assert_size_stride(primals_440, (768, ), (1, ))
    assert_size_stride(primals_441, (768, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_442, (768, ), (1, ))
    assert_size_stride(primals_443, (768, ), (1, ))
    assert_size_stride(primals_444, (768, ), (1, ))
    assert_size_stride(primals_445, (768, ), (1, ))
    assert_size_stride(primals_446, (2176, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_447, (2432, ), (1, ))
    assert_size_stride(primals_448, (2432, ), (1, ))
    assert_size_stride(primals_449, (2432, ), (1, ))
    assert_size_stride(primals_450, (2432, ), (1, ))
    assert_size_stride(primals_451, (768, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_452, (768, ), (1, ))
    assert_size_stride(primals_453, (768, ), (1, ))
    assert_size_stride(primals_454, (768, ), (1, ))
    assert_size_stride(primals_455, (768, ), (1, ))
    assert_size_stride(primals_456, (768, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_457, (768, ), (1, ))
    assert_size_stride(primals_458, (768, ), (1, ))
    assert_size_stride(primals_459, (768, ), (1, ))
    assert_size_stride(primals_460, (768, ), (1, ))
    assert_size_stride(primals_461, (2176, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_462, (2560, ), (1, ))
    assert_size_stride(primals_463, (2560, ), (1, ))
    assert_size_stride(primals_464, (2560, ), (1, ))
    assert_size_stride(primals_465, (2560, ), (1, ))
    assert_size_stride(primals_466, (768, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(primals_467, (768, ), (1, ))
    assert_size_stride(primals_468, (768, ), (1, ))
    assert_size_stride(primals_469, (768, ), (1, ))
    assert_size_stride(primals_470, (768, ), (1, ))
    assert_size_stride(primals_471, (768, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_472, (768, ), (1, ))
    assert_size_stride(primals_473, (768, ), (1, ))
    assert_size_stride(primals_474, (768, ), (1, ))
    assert_size_stride(primals_475, (768, ), (1, ))
    assert_size_stride(primals_476, (2176, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_477, (2688, ), (1, ))
    assert_size_stride(primals_478, (2688, ), (1, ))
    assert_size_stride(primals_479, (2688, ), (1, ))
    assert_size_stride(primals_480, (2688, ), (1, ))
    assert_size_stride(primals_481, (1000, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(primals_482, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 262144, grid=grid(262144), stream=stream0)
        del primals_6
        buf2 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.int8)
        buf4 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf6 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, batch_norm_1, relu_1, batch_norm_2, relu_2], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1.run(buf1, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, buf2, buf3, buf4, buf6, 65536, grid=grid(65536), stream=stream0)
        del primals_10
        del primals_15
        # Topologically Sorted Source Nodes: [x_s], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 288, 16, 16), (73728, 256, 16, 1))
        # Topologically Sorted Source Nodes: [x_in], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 96, 16, 16), (24576, 256, 16, 1))
        buf8 = empty_strided_cuda((4, 96, 16, 16), (24576, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf7, primals_17, primals_18, primals_19, primals_20, buf8, 98304, grid=grid(98304), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [x_in_1], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf9, (4, 96, 16, 16), (24576, 256, 16, 1))
        buf10 = empty_strided_cuda((4, 96, 16, 16), (24576, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_4, relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf9, primals_22, primals_23, primals_24, primals_25, buf10, 98304, grid=grid(98304), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [x_in_2], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 272, 16, 16), (69632, 256, 16, 1))
        buf12 = empty_strided_cuda((4, 304, 16, 16), (77824, 256, 16, 1), torch.float32)
        buf13 = empty_strided_cuda((4, 304, 16, 16), (77824, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_3, batch_norm_5, relu_5], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3.run(buf5, buf11, primals_27, primals_28, primals_29, primals_30, buf12, buf13, 311296, grid=grid(311296), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [x_in_4], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 96, 16, 16), (24576, 256, 16, 1))
        buf15 = empty_strided_cuda((4, 96, 16, 16), (24576, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_6, relu_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf14, primals_32, primals_33, primals_34, primals_35, buf15, 98304, grid=grid(98304), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [x_in_5], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf16, (4, 96, 16, 16), (24576, 256, 16, 1))
        buf17 = empty_strided_cuda((4, 96, 16, 16), (24576, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_7, relu_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf16, primals_37, primals_38, primals_39, primals_40, buf17, 98304, grid=grid(98304), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [x_in_6], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 272, 16, 16), (69632, 256, 16, 1))
        buf19 = empty_strided_cuda((4, 320, 16, 16), (81920, 256, 16, 1), torch.float32)
        buf20 = empty_strided_cuda((4, 320, 16, 16), (81920, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_7, batch_norm_8, relu_8], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4.run(buf5, buf11, buf18, primals_42, primals_43, primals_44, primals_45, buf19, buf20, 327680, grid=grid(327680), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [x_in_8], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 96, 16, 16), (24576, 256, 16, 1))
        buf22 = empty_strided_cuda((4, 96, 16, 16), (24576, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_9, relu_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf21, primals_47, primals_48, primals_49, primals_50, buf22, 98304, grid=grid(98304), stream=stream0)
        del primals_50
        # Topologically Sorted Source Nodes: [x_in_9], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf23, (4, 96, 16, 16), (24576, 256, 16, 1))
        buf24 = empty_strided_cuda((4, 96, 16, 16), (24576, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_10, relu_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf23, primals_52, primals_53, primals_54, primals_55, buf24, 98304, grid=grid(98304), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [x_in_10], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 272, 16, 16), (69632, 256, 16, 1))
        buf28 = empty_strided_cuda((4, 336, 16, 16), (86016, 256, 16, 1), torch.float32)
        buf26 = reinterpret_tensor(buf28, (4, 80, 16, 16), (86016, 256, 16, 1), 65536)  # alias
        # Topologically Sorted Source Nodes: [dense_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf5, buf11, buf18, buf25, buf26, 81920, grid=grid(81920), stream=stream0)
        buf27 = reinterpret_tensor(buf28, (4, 256, 16, 16), (86016, 256, 16, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid, resid_1, resid_2], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6.run(buf5, buf11, buf18, buf25, buf27, 262144, grid=grid(262144), stream=stream0)
        del buf11
        del buf18
        del buf25
        del buf5
        buf29 = empty_strided_cuda((4, 336, 16, 16), (86016, 256, 16, 1), torch.float32)
        buf31 = empty_strided_cuda((4, 336, 16, 16), (86016, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_11, relu_11, batch_norm_12, relu_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf28, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, buf29, buf31, 344064, grid=grid(344064), stream=stream0)
        del primals_60
        del primals_65
        # Topologically Sorted Source Nodes: [x_s_1], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_61, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 576, 8, 8), (36864, 64, 8, 1))
        # Topologically Sorted Source Nodes: [x_in_12], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 192, 16, 16), (49152, 256, 16, 1))
        buf33 = empty_strided_cuda((4, 192, 16, 16), (49152, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_13, relu_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf32, primals_67, primals_68, primals_69, primals_70, buf33, 196608, grid=grid(196608), stream=stream0)
        del primals_70
        # Topologically Sorted Source Nodes: [x_in_13], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_71, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf34, (4, 192, 8, 8), (12288, 64, 8, 1))
        buf35 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_14, relu_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf34, primals_72, primals_73, primals_74, primals_75, buf35, 49152, grid=grid(49152), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [x_in_14], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 544, 8, 8), (34816, 64, 8, 1))
        buf37 = empty_strided_cuda((4, 608, 8, 8), (38912, 64, 8, 1), torch.float32)
        buf38 = empty_strided_cuda((4, 608, 8, 8), (38912, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_15, batch_norm_15, relu_15], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10.run(buf30, buf36, primals_77, primals_78, primals_79, primals_80, buf37, buf38, 155648, grid=grid(155648), stream=stream0)
        del primals_80
        # Topologically Sorted Source Nodes: [x_in_16], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 192, 8, 8), (12288, 64, 8, 1))
        buf40 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_16, relu_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf39, primals_82, primals_83, primals_84, primals_85, buf40, 49152, grid=grid(49152), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [x_in_17], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf41, (4, 192, 8, 8), (12288, 64, 8, 1))
        buf42 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_17, relu_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf41, primals_87, primals_88, primals_89, primals_90, buf42, 49152, grid=grid(49152), stream=stream0)
        del primals_90
        # Topologically Sorted Source Nodes: [x_in_18], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 544, 8, 8), (34816, 64, 8, 1))
        buf44 = empty_strided_cuda((4, 640, 8, 8), (40960, 64, 8, 1), torch.float32)
        buf45 = empty_strided_cuda((4, 640, 8, 8), (40960, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_19, batch_norm_18, relu_18], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11.run(buf30, buf36, buf43, primals_92, primals_93, primals_94, primals_95, buf44, buf45, 163840, grid=grid(163840), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [x_in_20], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 192, 8, 8), (12288, 64, 8, 1))
        buf47 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_19, relu_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf46, primals_97, primals_98, primals_99, primals_100, buf47, 49152, grid=grid(49152), stream=stream0)
        del primals_100
        # Topologically Sorted Source Nodes: [x_in_21], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf48, (4, 192, 8, 8), (12288, 64, 8, 1))
        buf49 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_20, relu_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf48, primals_102, primals_103, primals_104, primals_105, buf49, 49152, grid=grid(49152), stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [x_in_22], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 544, 8, 8), (34816, 64, 8, 1))
        buf53 = empty_strided_cuda((4, 672, 8, 8), (43008, 64, 8, 1), torch.float32)
        buf51 = reinterpret_tensor(buf53, (4, 160, 8, 8), (43008, 64, 8, 1), 32768)  # alias
        # Topologically Sorted Source Nodes: [dense_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf30, buf36, buf43, buf50, buf51, 40960, grid=grid(40960), stream=stream0)
        buf52 = reinterpret_tensor(buf53, (4, 512, 8, 8), (43008, 64, 8, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid_3, resid_4, resid_5], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_13.run(buf30, buf36, buf43, buf50, buf52, 131072, grid=grid(131072), stream=stream0)
        del buf30
        del buf36
        del buf43
        del buf50
        buf54 = empty_strided_cuda((4, 672, 8, 8), (43008, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_21, relu_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf53, primals_107, primals_108, primals_109, primals_110, buf54, 172032, grid=grid(172032), stream=stream0)
        del primals_110
        # Topologically Sorted Source Nodes: [x_in_24], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 192, 8, 8), (12288, 64, 8, 1))
        buf56 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_22, relu_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf55, primals_112, primals_113, primals_114, primals_115, buf56, 49152, grid=grid(49152), stream=stream0)
        del primals_115
        # Topologically Sorted Source Nodes: [x_in_25], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf57, (4, 192, 8, 8), (12288, 64, 8, 1))
        buf58 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_23, relu_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf57, primals_117, primals_118, primals_119, primals_120, buf58, 49152, grid=grid(49152), stream=stream0)
        del primals_120
        # Topologically Sorted Source Nodes: [x_in_26], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 544, 8, 8), (34816, 64, 8, 1))
        buf60 = empty_strided_cuda((4, 704, 8, 8), (45056, 64, 8, 1), torch.float32)
        buf61 = empty_strided_cuda((4, 704, 8, 8), (45056, 64, 8, 1), torch.float32)
        buf63 = empty_strided_cuda((4, 704, 8, 8), (45056, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_27, batch_norm_24, relu_24, batch_norm_25, relu_25], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15.run(buf52, buf59, buf51, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, buf60, buf61, buf63, 180224, grid=grid(180224), stream=stream0)
        del buf59
        del primals_125
        del primals_130
        # Topologically Sorted Source Nodes: [x_s_2], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_126, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 1072, 4, 4), (17152, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_in_28], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 384, 8, 8), (24576, 64, 8, 1))
        buf65 = empty_strided_cuda((4, 384, 8, 8), (24576, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_26, relu_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf64, primals_132, primals_133, primals_134, primals_135, buf65, 98304, grid=grid(98304), stream=stream0)
        del primals_135
        # Topologically Sorted Source Nodes: [x_in_29], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_136, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf66, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf67 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_27, relu_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf66, primals_137, primals_138, primals_139, primals_140, buf67, 24576, grid=grid(24576), stream=stream0)
        del primals_140
        # Topologically Sorted Source Nodes: [x_in_30], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf69 = empty_strided_cuda((4, 1096, 4, 4), (17536, 16, 4, 1), torch.float32)
        buf70 = empty_strided_cuda((4, 1096, 4, 4), (17536, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_31, batch_norm_28, relu_28], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18.run(buf62, buf68, primals_142, primals_143, primals_144, primals_145, buf69, buf70, 70144, grid=grid(70144), stream=stream0)
        del primals_145
        # Topologically Sorted Source Nodes: [x_in_32], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf72 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_29, relu_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf71, primals_147, primals_148, primals_149, primals_150, buf72, 24576, grid=grid(24576), stream=stream0)
        del primals_150
        # Topologically Sorted Source Nodes: [x_in_33], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf73, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf74 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_30, relu_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf73, primals_152, primals_153, primals_154, primals_155, buf74, 24576, grid=grid(24576), stream=stream0)
        del primals_155
        # Topologically Sorted Source Nodes: [x_in_34], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf76 = empty_strided_cuda((4, 1120, 4, 4), (17920, 16, 4, 1), torch.float32)
        buf77 = empty_strided_cuda((4, 1120, 4, 4), (17920, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_35, batch_norm_31, relu_31], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf62, buf68, buf75, primals_157, primals_158, primals_159, primals_160, buf76, buf77, 71680, grid=grid(71680), stream=stream0)
        del primals_160
        # Topologically Sorted Source Nodes: [x_in_36], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf79 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_32, relu_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf78, primals_162, primals_163, primals_164, primals_165, buf79, 24576, grid=grid(24576), stream=stream0)
        del primals_165
        # Topologically Sorted Source Nodes: [x_in_37], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_166, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf80, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf81 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_33, relu_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf80, primals_167, primals_168, primals_169, primals_170, buf81, 24576, grid=grid(24576), stream=stream0)
        del primals_170
        # Topologically Sorted Source Nodes: [x_in_38], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf85 = empty_strided_cuda((4, 1144, 4, 4), (18304, 16, 4, 1), torch.float32)
        buf83 = reinterpret_tensor(buf85, (4, 120, 4, 4), (18304, 16, 4, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [dense_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf62, buf68, buf75, buf82, buf83, 7680, grid=grid(7680), stream=stream0)
        buf84 = reinterpret_tensor(buf85, (4, 1024, 4, 4), (18304, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid_7, resid_8, resid_9], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_21.run(buf62, buf68, buf75, buf82, buf84, 65536, grid=grid(65536), stream=stream0)
        del buf62
        del buf68
        del buf75
        del buf82
        buf86 = empty_strided_cuda((4, 1144, 4, 4), (18304, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_34, relu_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf85, primals_172, primals_173, primals_174, primals_175, buf86, 73216, grid=grid(73216), stream=stream0)
        del primals_175
        # Topologically Sorted Source Nodes: [x_in_40], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf88 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_35, relu_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf87, primals_177, primals_178, primals_179, primals_180, buf88, 24576, grid=grid(24576), stream=stream0)
        del primals_180
        # Topologically Sorted Source Nodes: [x_in_41], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf89, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf90 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_36, relu_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf89, primals_182, primals_183, primals_184, primals_185, buf90, 24576, grid=grid(24576), stream=stream0)
        del primals_185
        # Topologically Sorted Source Nodes: [x_in_42], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf92 = empty_strided_cuda((4, 1168, 4, 4), (18688, 16, 4, 1), torch.float32)
        buf93 = empty_strided_cuda((4, 1168, 4, 4), (18688, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_43, batch_norm_37, relu_37], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23.run(buf84, buf91, buf83, primals_187, primals_188, primals_189, primals_190, buf92, buf93, 74752, grid=grid(74752), stream=stream0)
        del primals_190
        # Topologically Sorted Source Nodes: [x_in_44], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_191, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf95 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_38, relu_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf94, primals_192, primals_193, primals_194, primals_195, buf95, 24576, grid=grid(24576), stream=stream0)
        del primals_195
        # Topologically Sorted Source Nodes: [x_in_45], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_196, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf96, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf97 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_39, relu_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf96, primals_197, primals_198, primals_199, primals_200, buf97, 24576, grid=grid(24576), stream=stream0)
        del primals_200
        # Topologically Sorted Source Nodes: [x_in_46], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf99 = empty_strided_cuda((4, 1192, 4, 4), (19072, 16, 4, 1), torch.float32)
        buf100 = empty_strided_cuda((4, 1192, 4, 4), (19072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_47, batch_norm_40, relu_40], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24.run(buf84, buf91, buf98, buf83, primals_202, primals_203, primals_204, primals_205, buf99, buf100, 76288, grid=grid(76288), stream=stream0)
        del primals_205
        # Topologically Sorted Source Nodes: [x_in_48], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf102 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_41, relu_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf101, primals_207, primals_208, primals_209, primals_210, buf102, 24576, grid=grid(24576), stream=stream0)
        del primals_210
        # Topologically Sorted Source Nodes: [x_in_49], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_211, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf103, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf104 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_42, relu_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf103, primals_212, primals_213, primals_214, primals_215, buf104, 24576, grid=grid(24576), stream=stream0)
        del primals_215
        # Topologically Sorted Source Nodes: [x_in_50], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf108 = empty_strided_cuda((4, 1216, 4, 4), (19456, 16, 4, 1), torch.float32)
        buf106 = reinterpret_tensor(buf108, (4, 192, 4, 4), (19456, 16, 4, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [dense_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_25.run(buf83, buf91, buf98, buf105, buf106, 12288, grid=grid(12288), stream=stream0)
        buf107 = reinterpret_tensor(buf108, (4, 1024, 4, 4), (19456, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid_10, resid_11, resid_12], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_26.run(buf84, buf91, buf98, buf105, buf107, 65536, grid=grid(65536), stream=stream0)
        del buf105
        del buf91
        del buf98
        buf109 = empty_strided_cuda((4, 1216, 4, 4), (19456, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_43, relu_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf108, primals_217, primals_218, primals_219, primals_220, buf109, 77824, grid=grid(77824), stream=stream0)
        del primals_220
        # Topologically Sorted Source Nodes: [x_in_52], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf111 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_44, relu_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf110, primals_222, primals_223, primals_224, primals_225, buf111, 24576, grid=grid(24576), stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [x_in_53], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf112, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf113 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_45, relu_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf112, primals_227, primals_228, primals_229, primals_230, buf113, 24576, grid=grid(24576), stream=stream0)
        del primals_230
        # Topologically Sorted Source Nodes: [x_in_54], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf115 = empty_strided_cuda((4, 1240, 4, 4), (19840, 16, 4, 1), torch.float32)
        buf116 = empty_strided_cuda((4, 1240, 4, 4), (19840, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_55, batch_norm_46, relu_46], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28.run(buf107, buf114, buf106, primals_232, primals_233, primals_234, primals_235, buf115, buf116, 79360, grid=grid(79360), stream=stream0)
        del primals_235
        # Topologically Sorted Source Nodes: [x_in_56], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf118 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_47, relu_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf117, primals_237, primals_238, primals_239, primals_240, buf118, 24576, grid=grid(24576), stream=stream0)
        del primals_240
        # Topologically Sorted Source Nodes: [x_in_57], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_241, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf119, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf120 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_48, relu_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf119, primals_242, primals_243, primals_244, primals_245, buf120, 24576, grid=grid(24576), stream=stream0)
        del primals_245
        # Topologically Sorted Source Nodes: [x_in_58], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf122 = empty_strided_cuda((4, 1264, 4, 4), (20224, 16, 4, 1), torch.float32)
        buf123 = empty_strided_cuda((4, 1264, 4, 4), (20224, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_59, batch_norm_49, relu_49], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29.run(buf107, buf114, buf121, buf106, primals_247, primals_248, primals_249, primals_250, buf122, buf123, 80896, grid=grid(80896), stream=stream0)
        del primals_250
        # Topologically Sorted Source Nodes: [x_in_60], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf125 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_50, relu_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf124, primals_252, primals_253, primals_254, primals_255, buf125, 24576, grid=grid(24576), stream=stream0)
        del primals_255
        # Topologically Sorted Source Nodes: [x_in_61], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf126, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf127 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_51, relu_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf126, primals_257, primals_258, primals_259, primals_260, buf127, 24576, grid=grid(24576), stream=stream0)
        del primals_260
        # Topologically Sorted Source Nodes: [x_in_62], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf131 = empty_strided_cuda((4, 1288, 4, 4), (20608, 16, 4, 1), torch.float32)
        buf129 = reinterpret_tensor(buf131, (4, 264, 4, 4), (20608, 16, 4, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [dense_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_30.run(buf106, buf114, buf121, buf128, buf129, 16896, grid=grid(16896), stream=stream0)
        buf130 = reinterpret_tensor(buf131, (4, 1024, 4, 4), (20608, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid_13, resid_14, resid_15], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_31.run(buf107, buf114, buf121, buf128, buf130, 65536, grid=grid(65536), stream=stream0)
        del buf114
        del buf121
        del buf128
        buf132 = empty_strided_cuda((4, 1288, 4, 4), (20608, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_52, relu_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf131, primals_262, primals_263, primals_264, primals_265, buf132, 82432, grid=grid(82432), stream=stream0)
        del primals_265
        # Topologically Sorted Source Nodes: [x_in_64], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_266, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf134 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_53, relu_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf133, primals_267, primals_268, primals_269, primals_270, buf134, 24576, grid=grid(24576), stream=stream0)
        del primals_270
        # Topologically Sorted Source Nodes: [x_in_65], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_271, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf135, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf136 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_54, relu_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf135, primals_272, primals_273, primals_274, primals_275, buf136, 24576, grid=grid(24576), stream=stream0)
        del primals_275
        # Topologically Sorted Source Nodes: [x_in_66], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_276, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf138 = empty_strided_cuda((4, 1312, 4, 4), (20992, 16, 4, 1), torch.float32)
        buf139 = empty_strided_cuda((4, 1312, 4, 4), (20992, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_67, batch_norm_55, relu_55], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33.run(buf130, buf137, buf129, primals_277, primals_278, primals_279, primals_280, buf138, buf139, 83968, grid=grid(83968), stream=stream0)
        del primals_280
        # Topologically Sorted Source Nodes: [x_in_68], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf141 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_56, relu_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf140, primals_282, primals_283, primals_284, primals_285, buf141, 24576, grid=grid(24576), stream=stream0)
        del primals_285
        # Topologically Sorted Source Nodes: [x_in_69], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_286, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf142, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf143 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_57, relu_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf142, primals_287, primals_288, primals_289, primals_290, buf143, 24576, grid=grid(24576), stream=stream0)
        del primals_290
        # Topologically Sorted Source Nodes: [x_in_70], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf145 = empty_strided_cuda((4, 1336, 4, 4), (21376, 16, 4, 1), torch.float32)
        buf146 = empty_strided_cuda((4, 1336, 4, 4), (21376, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_71, batch_norm_58, relu_58], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34.run(buf130, buf137, buf144, buf129, primals_292, primals_293, primals_294, primals_295, buf145, buf146, 85504, grid=grid(85504), stream=stream0)
        del primals_295
        # Topologically Sorted Source Nodes: [x_in_72], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_296, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf148 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_59, relu_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf147, primals_297, primals_298, primals_299, primals_300, buf148, 24576, grid=grid(24576), stream=stream0)
        del primals_300
        # Topologically Sorted Source Nodes: [x_in_73], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_301, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf149, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf150 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_60, relu_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf149, primals_302, primals_303, primals_304, primals_305, buf150, 24576, grid=grid(24576), stream=stream0)
        del primals_305
        # Topologically Sorted Source Nodes: [x_in_74], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_306, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf154 = empty_strided_cuda((4, 1360, 4, 4), (21760, 16, 4, 1), torch.float32)
        buf152 = reinterpret_tensor(buf154, (4, 336, 4, 4), (21760, 16, 4, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [dense_18], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf129, buf137, buf144, buf151, buf152, 21504, grid=grid(21504), stream=stream0)
        buf153 = reinterpret_tensor(buf154, (4, 1024, 4, 4), (21760, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid_16, resid_17, resid_18], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_36.run(buf130, buf137, buf144, buf151, buf153, 65536, grid=grid(65536), stream=stream0)
        del buf137
        del buf144
        del buf151
        buf155 = empty_strided_cuda((4, 1360, 4, 4), (21760, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_61, relu_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf154, primals_307, primals_308, primals_309, primals_310, buf155, 87040, grid=grid(87040), stream=stream0)
        del primals_310
        # Topologically Sorted Source Nodes: [x_in_76], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_311, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf157 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_62, relu_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf156, primals_312, primals_313, primals_314, primals_315, buf157, 24576, grid=grid(24576), stream=stream0)
        del primals_315
        # Topologically Sorted Source Nodes: [x_in_77], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_316, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf158, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf159 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_63, relu_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf158, primals_317, primals_318, primals_319, primals_320, buf159, 24576, grid=grid(24576), stream=stream0)
        del primals_320
        # Topologically Sorted Source Nodes: [x_in_78], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf161 = empty_strided_cuda((4, 1384, 4, 4), (22144, 16, 4, 1), torch.float32)
        buf162 = empty_strided_cuda((4, 1384, 4, 4), (22144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_79, batch_norm_64, relu_64], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38.run(buf153, buf160, buf152, primals_322, primals_323, primals_324, primals_325, buf161, buf162, 88576, grid=grid(88576), stream=stream0)
        del primals_325
        # Topologically Sorted Source Nodes: [x_in_80], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_326, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf164 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_65, relu_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf163, primals_327, primals_328, primals_329, primals_330, buf164, 24576, grid=grid(24576), stream=stream0)
        del primals_330
        # Topologically Sorted Source Nodes: [x_in_81], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_331, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf165, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf166 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_66, relu_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf165, primals_332, primals_333, primals_334, primals_335, buf166, 24576, grid=grid(24576), stream=stream0)
        del primals_335
        # Topologically Sorted Source Nodes: [x_in_82], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_336, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf168 = empty_strided_cuda((4, 1408, 4, 4), (22528, 16, 4, 1), torch.float32)
        buf169 = empty_strided_cuda((4, 1408, 4, 4), (22528, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_83, batch_norm_67, relu_67], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39.run(buf153, buf160, buf167, buf152, primals_337, primals_338, primals_339, primals_340, buf168, buf169, 90112, grid=grid(90112), stream=stream0)
        del primals_340
        # Topologically Sorted Source Nodes: [x_in_84], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_341, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf171 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_68, relu_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf170, primals_342, primals_343, primals_344, primals_345, buf171, 24576, grid=grid(24576), stream=stream0)
        del primals_345
        # Topologically Sorted Source Nodes: [x_in_85], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_346, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf172, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf173 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_69, relu_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf172, primals_347, primals_348, primals_349, primals_350, buf173, 24576, grid=grid(24576), stream=stream0)
        del primals_350
        # Topologically Sorted Source Nodes: [x_in_86], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_351, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf177 = empty_strided_cuda((4, 1432, 4, 4), (22912, 16, 4, 1), torch.float32)
        buf175 = reinterpret_tensor(buf177, (4, 408, 4, 4), (22912, 16, 4, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [dense_21], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf152, buf160, buf167, buf174, buf175, 26112, grid=grid(26112), stream=stream0)
        buf176 = reinterpret_tensor(buf177, (4, 1024, 4, 4), (22912, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid_19, resid_20, resid_21], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_41.run(buf153, buf160, buf167, buf174, buf176, 65536, grid=grid(65536), stream=stream0)
        del buf160
        del buf167
        del buf174
        buf178 = empty_strided_cuda((4, 1432, 4, 4), (22912, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_70, relu_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_42.run(buf177, primals_352, primals_353, primals_354, primals_355, buf178, 91648, grid=grid(91648), stream=stream0)
        del primals_355
        # Topologically Sorted Source Nodes: [x_in_88], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_356, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf180 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_71, relu_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf179, primals_357, primals_358, primals_359, primals_360, buf180, 24576, grid=grid(24576), stream=stream0)
        del primals_360
        # Topologically Sorted Source Nodes: [x_in_89], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_361, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf181, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf182 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_72, relu_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf181, primals_362, primals_363, primals_364, primals_365, buf182, 24576, grid=grid(24576), stream=stream0)
        del primals_365
        # Topologically Sorted Source Nodes: [x_in_90], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_366, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf184 = empty_strided_cuda((4, 1456, 4, 4), (23296, 16, 4, 1), torch.float32)
        buf185 = empty_strided_cuda((4, 1456, 4, 4), (23296, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_91, batch_norm_73, relu_73], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43.run(buf176, buf183, buf175, primals_367, primals_368, primals_369, primals_370, buf184, buf185, 93184, grid=grid(93184), stream=stream0)
        del primals_370
        # Topologically Sorted Source Nodes: [x_in_92], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_371, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf187 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_74, relu_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf186, primals_372, primals_373, primals_374, primals_375, buf187, 24576, grid=grid(24576), stream=stream0)
        del primals_375
        # Topologically Sorted Source Nodes: [x_in_93], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_376, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf188, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf189 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_75, relu_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf188, primals_377, primals_378, primals_379, primals_380, buf189, 24576, grid=grid(24576), stream=stream0)
        del primals_380
        # Topologically Sorted Source Nodes: [x_in_94], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_381, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf191 = empty_strided_cuda((4, 1480, 4, 4), (23680, 16, 4, 1), torch.float32)
        buf192 = empty_strided_cuda((4, 1480, 4, 4), (23680, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_95, batch_norm_76, relu_76], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44.run(buf176, buf183, buf190, buf175, primals_382, primals_383, primals_384, primals_385, buf191, buf192, 94720, grid=grid(94720), stream=stream0)
        del primals_385
        # Topologically Sorted Source Nodes: [x_in_96], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_386, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf194 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_77, relu_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf193, primals_387, primals_388, primals_389, primals_390, buf194, 24576, grid=grid(24576), stream=stream0)
        del primals_390
        # Topologically Sorted Source Nodes: [x_in_97], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, primals_391, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf195, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf196 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_78, relu_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf195, primals_392, primals_393, primals_394, primals_395, buf196, 24576, grid=grid(24576), stream=stream0)
        del primals_395
        # Topologically Sorted Source Nodes: [x_in_98], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_396, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf200 = empty_strided_cuda((4, 1504, 4, 4), (24064, 16, 4, 1), torch.float32)
        buf198 = reinterpret_tensor(buf200, (4, 480, 4, 4), (24064, 16, 4, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [dense_24], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_45.run(buf175, buf183, buf190, buf197, buf198, 30720, grid=grid(30720), stream=stream0)
        buf199 = reinterpret_tensor(buf200, (4, 1024, 4, 4), (24064, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid_22, resid_23, resid_24], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_46.run(buf176, buf183, buf190, buf197, buf199, 65536, grid=grid(65536), stream=stream0)
        del buf183
        del buf190
        del buf197
        buf201 = empty_strided_cuda((4, 1504, 4, 4), (24064, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_79, relu_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf200, primals_397, primals_398, primals_399, primals_400, buf201, 96256, grid=grid(96256), stream=stream0)
        del primals_400
        # Topologically Sorted Source Nodes: [x_in_100], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_401, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf203 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_80, relu_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf202, primals_402, primals_403, primals_404, primals_405, buf203, 24576, grid=grid(24576), stream=stream0)
        del primals_405
        # Topologically Sorted Source Nodes: [x_in_101], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_406, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf204, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf205 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_81, relu_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf204, primals_407, primals_408, primals_409, primals_410, buf205, 24576, grid=grid(24576), stream=stream0)
        del primals_410
        # Topologically Sorted Source Nodes: [x_in_102], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_411, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf207 = empty_strided_cuda((4, 1528, 4, 4), (24448, 16, 4, 1), torch.float32)
        buf208 = empty_strided_cuda((4, 1528, 4, 4), (24448, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_103, batch_norm_82, relu_82], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48.run(buf199, buf206, buf198, primals_412, primals_413, primals_414, primals_415, buf207, buf208, 97792, grid=grid(97792), stream=stream0)
        del primals_415
        # Topologically Sorted Source Nodes: [x_in_104], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_416, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf210 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_83, relu_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf209, primals_417, primals_418, primals_419, primals_420, buf210, 24576, grid=grid(24576), stream=stream0)
        del primals_420
        # Topologically Sorted Source Nodes: [x_in_105], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_421, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf211, (4, 384, 4, 4), (6144, 16, 4, 1))
        buf212 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_84, relu_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf211, primals_422, primals_423, primals_424, primals_425, buf212, 24576, grid=grid(24576), stream=stream0)
        del primals_425
        # Topologically Sorted Source Nodes: [x_in_106], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_426, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 1048, 4, 4), (16768, 16, 4, 1))
        buf214 = empty_strided_cuda((4, 1552, 4, 4), (24832, 16, 4, 1), torch.float32)
        buf215 = empty_strided_cuda((4, 1552, 4, 4), (24832, 16, 4, 1), torch.float32)
        buf217 = empty_strided_cuda((4, 1552, 4, 4), (24832, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_107, batch_norm_85, relu_85, batch_norm_86, relu_86], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_49.run(buf199, buf206, buf213, buf198, primals_427, primals_428, primals_429, primals_430, primals_432, primals_433, primals_434, primals_435, buf214, buf215, buf217, 99328, grid=grid(99328), stream=stream0)
        del buf206
        del buf213
        del primals_430
        del primals_435
        # Topologically Sorted Source Nodes: [x_s_3], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_431, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 2304, 2, 2), (9216, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_in_108], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_436, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 768, 4, 4), (12288, 16, 4, 1))
        buf219 = empty_strided_cuda((4, 768, 4, 4), (12288, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_87, relu_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf218, primals_437, primals_438, primals_439, primals_440, buf219, 49152, grid=grid(49152), stream=stream0)
        del primals_440
        # Topologically Sorted Source Nodes: [x_in_109], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_441, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf220, (4, 768, 2, 2), (3072, 4, 2, 1))
        buf221 = empty_strided_cuda((4, 768, 2, 2), (3072, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_88, relu_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf220, primals_442, primals_443, primals_444, primals_445, buf221, 12288, grid=grid(12288), stream=stream0)
        del primals_445
        # Topologically Sorted Source Nodes: [x_in_110], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_446, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 2176, 2, 2), (8704, 4, 2, 1))
        buf223 = empty_strided_cuda((4, 2432, 2, 2), (9728, 4, 2, 1), torch.float32)
        buf224 = empty_strided_cuda((4, 2432, 2, 2), (9728, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_111, batch_norm_89, relu_89], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52.run(buf216, buf222, primals_447, primals_448, primals_449, primals_450, buf223, buf224, 38912, grid=grid(38912), stream=stream0)
        del primals_450
        # Topologically Sorted Source Nodes: [x_in_112], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_451, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 768, 2, 2), (3072, 4, 2, 1))
        buf226 = empty_strided_cuda((4, 768, 2, 2), (3072, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_90, relu_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf225, primals_452, primals_453, primals_454, primals_455, buf226, 12288, grid=grid(12288), stream=stream0)
        del primals_455
        # Topologically Sorted Source Nodes: [x_in_113], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_456, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf227, (4, 768, 2, 2), (3072, 4, 2, 1))
        buf228 = empty_strided_cuda((4, 768, 2, 2), (3072, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_91, relu_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf227, primals_457, primals_458, primals_459, primals_460, buf228, 12288, grid=grid(12288), stream=stream0)
        del primals_460
        # Topologically Sorted Source Nodes: [x_in_114], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_461, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 2176, 2, 2), (8704, 4, 2, 1))
        buf230 = empty_strided_cuda((4, 2560, 2, 2), (10240, 4, 2, 1), torch.float32)
        buf231 = empty_strided_cuda((4, 2560, 2, 2), (10240, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_115, batch_norm_92, relu_92], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53.run(buf216, buf222, buf229, primals_462, primals_463, primals_464, primals_465, buf230, buf231, 40960, grid=grid(40960), stream=stream0)
        del primals_465
        # Topologically Sorted Source Nodes: [x_in_116], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_466, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 768, 2, 2), (3072, 4, 2, 1))
        buf233 = empty_strided_cuda((4, 768, 2, 2), (3072, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_93, relu_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf232, primals_467, primals_468, primals_469, primals_470, buf233, 12288, grid=grid(12288), stream=stream0)
        del primals_470
        # Topologically Sorted Source Nodes: [x_in_117], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_471, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf234, (4, 768, 2, 2), (3072, 4, 2, 1))
        buf235 = empty_strided_cuda((4, 768, 2, 2), (3072, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_94, relu_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf234, primals_472, primals_473, primals_474, primals_475, buf235, 12288, grid=grid(12288), stream=stream0)
        del primals_475
        # Topologically Sorted Source Nodes: [x_in_118], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_476, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 2176, 2, 2), (8704, 4, 2, 1))
        buf239 = empty_strided_cuda((4, 2688, 2, 2), (10752, 4, 2, 1), torch.float32)
        buf237 = reinterpret_tensor(buf239, (4, 640, 2, 2), (10752, 4, 2, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [dense_29], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_54.run(buf216, buf222, buf229, buf236, buf237, 10240, grid=grid(10240), stream=stream0)
        buf238 = reinterpret_tensor(buf239, (4, 2048, 2, 2), (10752, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid_27, resid_28, resid_29], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_55.run(buf216, buf222, buf229, buf236, buf238, 32768, grid=grid(32768), stream=stream0)
        del buf216
        del buf222
        del buf229
        del buf236
        buf240 = empty_strided_cuda((4, 2688, 2, 2), (10752, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_95, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_56.run(buf239, primals_477, primals_478, primals_479, primals_480, buf240, 43008, grid=grid(43008), stream=stream0)
        del primals_480
        buf241 = empty_strided_cuda((4, 2688, 1, 1), (2688, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_57.run(buf240, buf241, 10752, grid=grid(10752), stream=stream0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_481, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 1000, 1, 1), (1000, 1, 1, 1))
        buf243 = reinterpret_tensor(buf242, (4, 1000, 1, 1), (1000, 1, 4000, 4000), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_58.run(buf243, primals_482, 4000, grid=grid(4000), stream=stream0)
        del primals_482
    return (reinterpret_tensor(buf243, (4, 1000), (1000, 1), 0), primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_29, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_46, primals_47, primals_48, primals_49, primals_51, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_59, primals_61, primals_62, primals_63, primals_64, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_86, primals_87, primals_88, primals_89, primals_91, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_99, primals_101, primals_102, primals_103, primals_104, primals_106, primals_107, primals_108, primals_109, primals_111, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, primals_119, primals_121, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_129, primals_131, primals_132, primals_133, primals_134, primals_136, primals_137, primals_138, primals_139, primals_141, primals_142, primals_143, primals_144, primals_146, primals_147, primals_148, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_167, primals_168, primals_169, primals_171, primals_172, primals_173, primals_174, primals_176, primals_177, primals_178, primals_179, primals_181, primals_182, primals_183, primals_184, primals_186, primals_187, primals_188, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_202, primals_203, primals_204, primals_206, primals_207, primals_208, primals_209, primals_211, primals_212, primals_213, primals_214, primals_216, primals_217, primals_218, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, primals_229, primals_231, primals_232, primals_233, primals_234, primals_236, primals_237, primals_238, primals_239, primals_241, primals_242, primals_243, primals_244, primals_246, primals_247, primals_248, primals_249, primals_251, primals_252, primals_253, primals_254, primals_256, primals_257, primals_258, primals_259, primals_261, primals_262, primals_263, primals_264, primals_266, primals_267, primals_268, primals_269, primals_271, primals_272, primals_273, primals_274, primals_276, primals_277, primals_278, primals_279, primals_281, primals_282, primals_283, primals_284, primals_286, primals_287, primals_288, primals_289, primals_291, primals_292, primals_293, primals_294, primals_296, primals_297, primals_298, primals_299, primals_301, primals_302, primals_303, primals_304, primals_306, primals_307, primals_308, primals_309, primals_311, primals_312, primals_313, primals_314, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_326, primals_327, primals_328, primals_329, primals_331, primals_332, primals_333, primals_334, primals_336, primals_337, primals_338, primals_339, primals_341, primals_342, primals_343, primals_344, primals_346, primals_347, primals_348, primals_349, primals_351, primals_352, primals_353, primals_354, primals_356, primals_357, primals_358, primals_359, primals_361, primals_362, primals_363, primals_364, primals_366, primals_367, primals_368, primals_369, primals_371, primals_372, primals_373, primals_374, primals_376, primals_377, primals_378, primals_379, primals_381, primals_382, primals_383, primals_384, primals_386, primals_387, primals_388, primals_389, primals_391, primals_392, primals_393, primals_394, primals_396, primals_397, primals_398, primals_399, primals_401, primals_402, primals_403, primals_404, primals_406, primals_407, primals_408, primals_409, primals_411, primals_412, primals_413, primals_414, primals_416, primals_417, primals_418, primals_419, primals_421, primals_422, primals_423, primals_424, primals_426, primals_427, primals_428, primals_429, primals_431, primals_432, primals_433, primals_434, primals_436, primals_437, primals_438, primals_439, primals_441, primals_442, primals_443, primals_444, primals_446, primals_447, primals_448, primals_449, primals_451, primals_452, primals_453, primals_454, primals_456, primals_457, primals_458, primals_459, primals_461, primals_462, primals_463, primals_464, primals_466, primals_467, primals_468, primals_469, primals_471, primals_472, primals_473, primals_474, primals_476, primals_477, primals_478, primals_479, primals_481, buf0, buf1, buf2, buf3, buf4, buf6, buf7, buf8, buf9, buf10, buf12, buf13, buf14, buf15, buf16, buf17, buf19, buf20, buf21, buf22, buf23, buf24, buf28, buf29, buf31, buf32, buf33, buf34, buf35, buf37, buf38, buf39, buf40, buf41, buf42, buf44, buf45, buf46, buf47, buf48, buf49, buf53, buf54, buf55, buf56, buf57, buf58, buf60, buf61, buf63, buf64, buf65, buf66, buf67, buf69, buf70, buf71, buf72, buf73, buf74, buf76, buf77, buf78, buf79, buf80, buf81, buf85, buf86, buf87, buf88, buf89, buf90, buf92, buf93, buf94, buf95, buf96, buf97, buf99, buf100, buf101, buf102, buf103, buf104, buf108, buf109, buf110, buf111, buf112, buf113, buf115, buf116, buf117, buf118, buf119, buf120, buf122, buf123, buf124, buf125, buf126, buf127, buf131, buf132, buf133, buf134, buf135, buf136, buf138, buf139, buf140, buf141, buf142, buf143, buf145, buf146, buf147, buf148, buf149, buf150, buf154, buf155, buf156, buf157, buf158, buf159, buf161, buf162, buf163, buf164, buf165, buf166, buf168, buf169, buf170, buf171, buf172, buf173, buf177, buf178, buf179, buf180, buf181, buf182, buf184, buf185, buf186, buf187, buf188, buf189, buf191, buf192, buf193, buf194, buf195, buf196, buf200, buf201, buf202, buf203, buf204, buf205, buf207, buf208, buf209, buf210, buf211, buf212, buf214, buf215, buf217, buf218, buf219, buf220, buf221, buf223, buf224, buf225, buf226, buf227, buf228, buf230, buf231, buf232, buf233, buf234, buf235, buf239, buf240, buf241, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((288, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((96, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((272, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((96, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((272, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((96, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((96, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((272, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((576, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((192, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((192, 6, 3, 3), (54, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((544, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((192, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((192, 6, 3, 3), (54, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((544, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((192, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((192, 6, 3, 3), (54, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((544, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((192, 6, 3, 3), (54, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((544, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1072, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((384, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, 1096, 1, 1), (1096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((1120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, 1120, 1, 1), (1120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((384, 1144, 1, 1), (1144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((1168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((384, 1168, 1, 1), (1168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((384, 1192, 1, 1), (1192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((384, 1216, 1, 1), (1216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((1240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((1240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((384, 1240, 1, 1), (1240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((1264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((1264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((384, 1264, 1, 1), (1264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((1288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((384, 1288, 1, 1), (1288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((1312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((1312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((384, 1312, 1, 1), (1312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((384, 1336, 1, 1), (1336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((1360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((384, 1360, 1, 1), (1360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((1384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((1384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((384, 1384, 1, 1), (1384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((384, 1408, 1, 1), (1408, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((1432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((1432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((1432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((1432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((384, 1432, 1, 1), (1432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((1456, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((1456, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((1456, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((1456, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((384, 1456, 1, 1), (1456, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((1480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((1480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((1480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((384, 1480, 1, 1), (1480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((1504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((1504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((1504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((1504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((384, 1504, 1, 1), (1504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((1528, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((1528, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((1528, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1528, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((384, 1528, 1, 1), (1528, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((384, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((1048, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((1552, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((1552, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((1552, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((1552, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((2304, 1552, 1, 1), (1552, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((1552, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((1552, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((1552, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((1552, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((768, 1552, 1, 1), (1552, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((768, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((2176, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((768, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((768, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((2176, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((768, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((768, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((2176, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((1000, 2688, 1, 1), (2688, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
