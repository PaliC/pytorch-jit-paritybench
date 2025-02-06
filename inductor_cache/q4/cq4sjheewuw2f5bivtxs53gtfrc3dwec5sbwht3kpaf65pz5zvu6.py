# AOT ID: ['29_forward']
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


# kernel path: inductor_cache/5m/c5maepkiwsp7xrhsjzhibh6owtkoxcgtmbadyxuws6jxamdhttoq.py
# Topologically Sorted Source Nodes: [x_1, x_], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_ => relu
#   x_1 => add_1, mul_1, mul_2, sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=6] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
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
    tmp4 = 1e-05
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


# kernel path: inductor_cache/ko/ckoidgcxmfqmhvevmbelljzdwj3e3nm2osp7fw7hcc3wlqsfqj26.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_2 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_1 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x3 = xindex // 16
    x4 = xindex
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
    tmp11 = tl.load(in_ptr0 + ((-33) + 2*x0 + 64*x3), tmp10, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-32) + 2*x0 + 64*x3), tmp16, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-31) + 2*x0 + 64*x3), tmp23, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 64*x3), tmp30, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 64*x3), tmp33, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x3), tmp36, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (31 + 2*x0 + 64*x3), tmp43, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x3), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x3), tmp49, eviction_policy='evict_last', other=float("-inf"))
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
    tl.store(out_ptr0 + (x4), tmp51, None)
    tl.store(out_ptr1 + (x4), tmp76, None)
''', device_str='cuda')


# kernel path: inductor_cache/45/c45te3gf6p2uyvfymkpk7k62e32m3hvl55qijfda7nrtq2ax5qyt.py
# Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_1 => add_3, mul_4, mul_5, sub_1
#   out_2 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/ja/cjafr7me3ajr66xrgqm5sxemqhlnvgutia3imfdgj6jw26czabyw.py
# Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_4 => add_5, mul_7, mul_8, sub_2
#   out_5 => add_6
#   out_6 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %getitem), kwargs = {})
#   %relu_2 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/d3/cd3ffxgzg4teskulrubddpr27rm3h6pqt4pcyt6yuosk2bmdtfbp.py
# Topologically Sorted Source Nodes: [out_18, out_19, out_20, x_48, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   out_18 => add_15, mul_19, mul_20, sub_6
#   out_19 => add_16
#   out_20 => relu_6
#   x_48 => cat_4
#   x_92 => cat_8
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %relu_4), kwargs = {})
#   %relu_6 : [num_users=7] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_6, %relu_38, %_unsafe_index_4], 1), kwargs = {})
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_6, %relu_38, %relu_47, %_unsafe_index_8], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 256) % 64)
    x2 = xindex // 16384
    x3 = (xindex % 16384)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x4), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x4), tmp19, None)
    tl.store(out_ptr1 + (x3 + 65536*x2), tmp19, None)
    tl.store(out_ptr2 + (x3 + 131072*x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/er/certzuodxozfkdpefnxejttid562pglqkq52cpp5swahq7asaeee.py
# Topologically Sorted Source Nodes: [out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_22 => add_18, mul_22, mul_23, sub_7
#   out_23 => relu_7
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_18,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/rr/crrgza5oh2rgf6fbfku5kf37gmmyr4kw7ggyzqts4kqe6xnjv4az.py
# Topologically Sorted Source Nodes: [out_25, input_2, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_22, mul_28, mul_29, sub_9
#   out_25 => add_20, mul_25, mul_26, sub_8
#   out_26 => add_23
#   out_27 => relu_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %add_22), kwargs = {})
#   %relu_8 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/qv/cqvpgvvqincvxqwgf3nzq4nfoxky6zddgsyimkmd66vewbc4lgvn.py
# Topologically Sorted Source Nodes: [out_32, out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_32 => add_27, mul_34, mul_35, sub_11
#   out_33 => add_28
#   out_34 => relu_10
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %relu_8), kwargs = {})
#   %relu_10 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/22/c22jk5jr33crrrqbjfiwjyjenbho6lvwlgucdforzzqpugh57xem.py
# Topologically Sorted Source Nodes: [out_46, out_47, out_48, x_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   out_46 => add_37, mul_46, mul_47, sub_15
#   out_47 => add_38
#   out_48 => relu_14
#   x_81 => cat_7
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %relu_12), kwargs = {})
#   %relu_14 : [num_users=6] = call_function[target=torch.ops.aten.relu.default](args = (%add_38,), kwargs = {})
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_14, %relu_44, %_unsafe_index_7], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 64) % 128)
    x2 = xindex // 8192
    x3 = (xindex % 8192)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x4), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x4), tmp19, None)
    tl.store(out_ptr1 + (x3 + 32768*x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ym/cymsjesdybflihrq6dnuscfkmh7nnr3ky5kozrhbtrcbfjex2snk.py
# Topologically Sorted Source Nodes: [out_50, out_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_50 => add_40, mul_49, mul_50, sub_16
#   out_51 => relu_15
# Graph fragment:
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_40,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/lu/clujdi25gv2eivcek2zsoxz3unvjz2p7me42feagszichlkyfuts.py
# Topologically Sorted Source Nodes: [out_53, input_4, out_54, out_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_44, mul_55, mul_56, sub_18
#   out_53 => add_42, mul_52, mul_53, sub_17
#   out_54 => add_45
#   out_55 => relu_16
# Graph fragment:
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_42, %add_44), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_45,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/ew/cewvcqmqkpwqcee4ted2gzzev7cdje5y57geglep6il55sots7sd.py
# Topologically Sorted Source Nodes: [out_60, out_61, out_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_60 => add_49, mul_61, mul_62, sub_20
#   out_61 => add_50
#   out_62 => relu_18
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %relu_16), kwargs = {})
#   %relu_18 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_50,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/l5/cl5cixs7psw75u4l2siqhv52wndyisbt7jrfyo66jylr7hb3zeyp.py
# Topologically Sorted Source Nodes: [out_92, out_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_92 => add_72, mul_88, mul_89, sub_29
#   out_93 => relu_27
# Graph fragment:
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_237), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_239), kwargs = {})
#   %relu_27 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_72,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/mb/cmb7ytdnaipsw7o7bqsdtaqeccvhsn7av447z55dwy2pocawnoui.py
# Topologically Sorted Source Nodes: [out_95, input_6, out_96, out_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_76, mul_94, mul_95, sub_31
#   out_95 => add_74, mul_91, mul_92, sub_30
#   out_96 => add_77
#   out_97 => relu_28
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_249), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %unsqueeze_253), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %unsqueeze_255), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_74, %add_76), kwargs = {})
#   %relu_28 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_77,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/zb/czbdust6uzckqh45cd6dgm5o75gv7fjwn7u65oyk4pz3m7fypt5l.py
# Topologically Sorted Source Nodes: [out_102, out_103, out_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_102 => add_81, mul_100, mul_101, sub_33
#   out_103 => add_82
#   out_104 => relu_30
# Graph fragment:
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_81, %relu_28), kwargs = {})
#   %relu_30 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_82,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ms/cmssilr5g72s4z74xiyvmh6btkvmqvpdw54wgtajj3vznzi6pih5.py
# Topologically Sorted Source Nodes: [out_109, out_110, out_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_109 => add_86, mul_106, mul_107, sub_35
#   out_110 => add_87
#   out_111 => relu_32
# Graph fragment:
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_281), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %unsqueeze_285), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_107, %unsqueeze_287), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_86, %relu_30), kwargs = {})
#   %relu_32 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_87,), kwargs = {})
#   %le_30 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_32, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = 0.0
    tmp21 = tmp19 <= tmp20
    tl.store(out_ptr0 + (x3), tmp19, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/sq/csq5mwjswdx6bxbaclgpsabgjundvug2qc3g274k6j5oi46fko2f.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_3 => add_88, add_89, convert_element_type_72, convert_element_type_73, iota, mul_108, mul_109
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_108, 0), kwargs = {})
#   %convert_element_type_72 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_88, torch.float32), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_72, 0.0), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, 0.5), kwargs = {})
#   %convert_element_type_73 : [num_users=6] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_109, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_16 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_16(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci45au5o5jx5yxscpsoblv7gg5rmkpl55ky4y76zuzbbn645pwof.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_4 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu, %_unsafe_index], 1), kwargs = {})
triton_poi_fused_cat_17 = async_compile.triton('triton_poi_fused_cat_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 128)
    x3 = xindex // 131072
    x4 = (xindex % 1024)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 1024*(x2) + 65536*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([XBLOCK], 16, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tmp14 = tl.load(in_ptr1 + (x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 + tmp10
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr2 + (tmp17 + 16*tmp13 + 256*((-64) + x2) + 16384*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tl.store(out_ptr0 + (x5), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/we/cwe3zkppe6vua34md7kgiigal4rncdrh5ixjkrmnbizahcdi6bjy.py
# Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_6 => add_93, mul_113, mul_114, sub_36
#   x_7 => relu_33
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_290), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_292), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %unsqueeze_294), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %unsqueeze_296), kwargs = {})
#   %relu_33 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_93,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/tz/ctzpp6a4ox677o5doqnaalin2jff5emsocz7xui5ulcy7ptgfwy7.py
# Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_14 => add_98, add_99, convert_element_type_82, convert_element_type_83, iota_2, mul_121, mul_122
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, 0), kwargs = {})
#   %convert_element_type_82 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_98, torch.float32), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_82, 0.0), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_99, 0.5), kwargs = {})
#   %convert_element_type_83 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_122, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_19 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_19', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_19(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jm/cjmf3zh4kzdltaupnfkihtm7pk5v6fnihpnogeqjwhudl4qr6wyc.py
# Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_15 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_6, %_unsafe_index_1], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 192)
    x3 = xindex // 49152
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 16384*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([XBLOCK], 8, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tmp14 = tl.load(in_ptr1 + (x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 + tmp10
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr2 + (tmp17 + 8*tmp13 + 64*((-64) + x2) + 8192*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tl.store(out_ptr0 + (x5), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/7a/c7a6x2ivqcc44lmn2ggq3o24zdrvegrrwmgifj7kdpwgtzjqxfpj.py
# Topologically Sorted Source Nodes: [x_17, x_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_17 => add_103, mul_126, mul_127, sub_39
#   x_18 => relu_36
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_315), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_317), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_126, %unsqueeze_319), kwargs = {})
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_127, %unsqueeze_321), kwargs = {})
#   %relu_36 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_103,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/65/c65pv572fzdntbe6fi2tbnvffrfdj2gwvo6wgi4b6vkiodg25bnh.py
# Topologically Sorted Source Nodes: [x_23, x_24, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   x_23 => add_107, mul_132, mul_133, sub_41
#   x_24 => relu_38
#   x_92 => cat_8
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_331), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_333), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_132, %unsqueeze_335), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_133, %unsqueeze_337), kwargs = {})
#   %relu_38 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_6, %relu_38, %relu_47, %_unsafe_index_8], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 256) % 64)
    x2 = xindex // 16384
    x3 = (xindex % 16384)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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
    tl.store(out_ptr0 + (x3 + 65536*x2), tmp17, None)
    tl.store(out_ptr1 + (x3 + 131072*x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/mm/cmmniqazlkeexthml3sljitxy27qzhg4znqnk2otgl4qphxk4abe.py
# Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_26 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu, %relu_35, %_unsafe_index_2], 1), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 192)
    x3 = xindex // 196608
    x4 = (xindex % 1024)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 1024*(x2) + 65536*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + 1024*((-64) + x2) + 65536*x3), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x1), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full([XBLOCK], 16, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tmp19 = tl.load(in_ptr2 + (x0), tmp11, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp19 + tmp15
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr3 + (tmp22 + 16*tmp18 + 256*((-128) + x2) + 65536*x3), tmp11, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp9, tmp10, tmp23)
    tmp25 = tl.where(tmp4, tmp5, tmp24)
    tl.store(out_ptr0 + (x5), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/4f/c4fpordyr3khyjfxlktvv56ya4onsgillejrybbiqreh4r6lvfsi.py
# Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_36 => add_118, add_119, convert_element_type_102, convert_element_type_103, iota_6, mul_147, mul_148
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_147, 0), kwargs = {})
#   %convert_element_type_102 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_118, torch.float32), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_102, 0.0), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_119, 0.5), kwargs = {})
#   %convert_element_type_103 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_148, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_24 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_24(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kh/ckhgxibp5rzqlsm2rkanfbm6evnzuflsd2st3pf27j2gc3tb76s7.py
# Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_37 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_14, %_unsafe_index_3], 1), kwargs = {})
triton_poi_fused_cat_25 = async_compile.triton('triton_poi_fused_cat_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 384)
    x3 = xindex // 24576
    x4 = (xindex % 64)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 64*(x2) + 8192*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 384, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([XBLOCK], 4, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tmp14 = tl.load(in_ptr1 + (x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 + tmp10
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr2 + (tmp17 + 4*tmp13 + 16*((-128) + x2) + 4096*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tl.store(out_ptr0 + (x5), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/qa/cqal2lt43brvoahhitkt7xwqzlqafddl3nz6d3ktcqpqwqpraxrb.py
# Topologically Sorted Source Nodes: [x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_39 => add_123, mul_152, mul_153, sub_45
#   x_40 => relu_42
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_365), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_367), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_152, %unsqueeze_369), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_153, %unsqueeze_371), kwargs = {})
#   %relu_42 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_123,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/of/cofm63l7vsttjz45oticu6qflcssx4as3whiizgw2w5u2z4a4vbd.py
# Topologically Sorted Source Nodes: [x_45, x_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_45 => add_127, mul_158, mul_159, sub_47
#   x_46 => relu_44
# Graph fragment:
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_47, %unsqueeze_381), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_383), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_158, %unsqueeze_385), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_159, %unsqueeze_387), kwargs = {})
#   %relu_44 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_127,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    x2 = xindex // 8192
    x4 = (xindex % 8192)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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
    tl.store(out_ptr0 + (x4 + 32768*x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/6x/c6xzaojxoyc4mxhvx4eeq57ekngotjgpe6br5mgz5z7hnw4qc4wb.py
# Topologically Sorted Source Nodes: [x_47], Original ATen: [aten._unsafe_index]
# Source node to ATen node mapping:
#   x_47 => _unsafe_index_4
# Graph fragment:
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_44, [None, None, %unsqueeze_313, %convert_element_type_83]), kwargs = {})
triton_poi_fused__unsafe_index_28 = async_compile.triton('triton_poi_fused__unsafe_index_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_28(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = ((xindex // 256) % 128)
    x3 = xindex // 32768
    x5 = (xindex % 32768)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 8*tmp4 + 64*x2 + 32768*x3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + 65536*x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/ku/ckuhp6sldrs4mr3wyqgnnr42ousa7se2mpcbh6lbo2hacrizl5se.py
# Topologically Sorted Source Nodes: [x_56, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_56 => add_137, mul_171, mul_172, sub_50
#   x_57 => relu_47
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_406), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_408), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_171, %unsqueeze_410), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_172, %unsqueeze_412), kwargs = {})
#   %relu_47 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_137,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 128)
    x2 = xindex // 32768
    x4 = (xindex % 32768)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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
    tl.store(out_ptr0 + (x4 + 131072*x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/vb/cvbirwga6cgsqos6v3wiiy6nbmfvhkqaqd6p34anaqwhocuilozc.py
# Topologically Sorted Source Nodes: [x_59], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_59 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu, %relu_35, %relu_41, %_unsafe_index_5], 1), kwargs = {})
triton_poi_fused_cat_30 = async_compile.triton('triton_poi_fused_cat_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 320)
    x3 = xindex // 327680
    x4 = (xindex % 1024)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 1024*(x2) + 65536*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + 1024*((-64) + x2) + 65536*x3), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x4 + 1024*((-128) + x2) + 65536*x3), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 320, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x1), tmp16, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full([XBLOCK], 16, tl.int32)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp19 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp19)
    tmp24 = tl.load(in_ptr3 + (x0), tmp16, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp20
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tmp28 = tl.load(in_ptr4 + (tmp27 + 16*tmp23 + 256*((-192) + x2) + 131072*x3), tmp16, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.where(tmp14, tmp15, tmp28)
    tmp30 = tl.where(tmp9, tmp10, tmp29)
    tmp31 = tl.where(tmp4, tmp5, tmp30)
    tl.store(out_ptr0 + (x5), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/xo/cxolmpxxxgu7k7g6eu57znlwty6cwqmyeidovvxkmf762ecgabjg.py
# Topologically Sorted Source Nodes: [x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_67 => add_147, mul_184, mul_185, sub_53
#   x_68 => relu_50
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_431), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_433), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_435), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_437), kwargs = {})
#   %relu_50 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_147,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/g3/cg3fb66ixzevjfvkh4ko2c6dkpbr5mbzl6idr5tih4owmpz2zp3o.py
# Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_69 => add_148, add_149, convert_element_type_132, convert_element_type_133, iota_12, mul_186, mul_187
# Graph fragment:
#   %iota_12 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_12, 1), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_186, 0), kwargs = {})
#   %convert_element_type_132 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_148, torch.float32), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_132, 0.0), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_149, 0.5), kwargs = {})
#   %convert_element_type_133 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_187, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_32 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_32', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_32(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5g/c5goy7f2pxmdvds2ct7cws7w65jcu7a3iff7qt7a7csebpu3h5cp.py
# Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_70 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_26, %_unsafe_index_6], 1), kwargs = {})
triton_poi_fused_cat_33 = async_compile.triton('triton_poi_fused_cat_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_33(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 16) % 768)
    x3 = xindex // 12288
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 4096*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 768, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([XBLOCK], 2, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tmp14 = tl.load(in_ptr1 + (x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 + tmp10
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr2 + (tmp17 + 2*tmp13 + 4*((-256) + x2) + 2048*x3), tmp6, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tl.store(out_ptr0 + (x5), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ht/chtb764mc7xrlhigayrtirswcihlm7o2yebutkzuntlllxwnba5x.py
# Topologically Sorted Source Nodes: [x_72, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_72 => add_153, mul_191, mul_192, sub_54
#   x_73 => relu_51
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_54, %unsqueeze_440), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_442), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_191, %unsqueeze_444), kwargs = {})
#   %add_153 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_192, %unsqueeze_446), kwargs = {})
#   %relu_51 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_153,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/rx/crxqhfy72cdpcimopi4n3gms45cpvtie3w66opgahk52mhvvowft.py
# Topologically Sorted Source Nodes: [x_78, x_79, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   x_78 => add_157, mul_197, mul_198, sub_56
#   x_79 => relu_53
#   x_80 => _unsafe_index_7
# Graph fragment:
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_56, %unsqueeze_456), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_458), kwargs = {})
#   %mul_198 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_197, %unsqueeze_460), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_198, %unsqueeze_462), kwargs = {})
#   %relu_53 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_157,), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_53, [None, None, %unsqueeze_363, %convert_element_type_103]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex // 64
    x2 = ((xindex // 64) % 256)
    x3 = xindex // 16384
    x6 = (xindex % 16384)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 4*tmp4 + 16*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tl.store(out_ptr0 + (x6 + 32768*x3), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/os/cossk4x7jv2xjowrlcn4k4ere73ppew3jcqi5ziwac5343ca5off.py
# Topologically Sorted Source Nodes: [x_89, x_90, x_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   x_89 => add_167, mul_210, mul_211, sub_59
#   x_90 => relu_56
#   x_91 => _unsafe_index_8
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_59, %unsqueeze_481), kwargs = {})
#   %mul_210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_483), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_210, %unsqueeze_485), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_211, %unsqueeze_487), kwargs = {})
#   %relu_56 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_167,), kwargs = {})
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_56, [None, None, %unsqueeze_313, %convert_element_type_83]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex // 256
    x2 = ((xindex // 256) % 256)
    x3 = xindex // 65536
    x6 = (xindex % 65536)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 8*tmp4 + 64*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tl.store(out_ptr0 + (x6 + 131072*x3), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/yj/cyjwfunjhnygerzlluxcft6adwviyltuiurqs7anrwqonir4qeau.py
# Topologically Sorted Source Nodes: [x_94, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_94 => add_173, mul_217, mul_218, sub_60
#   x_95 => relu_57
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_60, %unsqueeze_490), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_492), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_494), kwargs = {})
#   %add_173 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_496), kwargs = {})
#   %relu_57 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_173,), kwargs = {})
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
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/gq/cgqmizeac2xltiwhjelb52jyy7pujdt5adaaxmuigp44knihnmbh.py
# Topologically Sorted Source Nodes: [x_103], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_103 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu, %relu_35, %relu_41, %relu_50, %_unsafe_index_9], 1), kwargs = {})
triton_poi_fused_cat_38 = async_compile.triton('triton_poi_fused_cat_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 576)
    x3 = xindex // 589824
    x4 = (xindex % 1024)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 1024*(x2) + 65536*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + 1024*((-64) + x2) + 65536*x3), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x4 + 1024*((-128) + x2) + 65536*x3), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 320, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x4 + 1024*((-192) + x2) + 131072*x3), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 576, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (x1), tmp21, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full([XBLOCK], 16, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tmp29 = tl.load(in_ptr4 + (x0), tmp21, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp25
    tmp31 = tmp29 < 0
    tmp32 = tl.where(tmp31, tmp30, tmp29)
    tmp33 = tl.load(in_ptr5 + (tmp32 + 16*tmp28 + 256*((-320) + x2) + 65536*x3), tmp21, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + ((-320) + x2), tmp21, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp33 - tmp34
    tmp36 = tl.load(in_ptr7 + ((-320) + x2), tmp21, eviction_policy='evict_last', other=0.0)
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.sqrt(tmp38)
    tmp40 = tl.full([1], 1, tl.int32)
    tmp41 = tmp40 / tmp39
    tmp42 = 1.0
    tmp43 = tmp41 * tmp42
    tmp44 = tmp35 * tmp43
    tmp45 = tl.load(in_ptr8 + ((-320) + x2), tmp21, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 * tmp45
    tmp47 = tl.load(in_ptr9 + ((-320) + x2), tmp21, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 + tmp47
    tmp49 = tl.full([1], 0, tl.int32)
    tmp50 = triton_helpers.maximum(tmp49, tmp48)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp21, tmp50, tmp51)
    tmp53 = tl.where(tmp19, tmp20, tmp52)
    tmp54 = tl.where(tmp14, tmp15, tmp53)
    tmp55 = tl.where(tmp9, tmp10, tmp54)
    tmp56 = tl.where(tmp4, tmp5, tmp55)
    tl.store(out_ptr0 + (x5), tmp56, None)
''', device_str='cuda')


# kernel path: inductor_cache/te/ctee4smcdz4qoxwvkpqxiuvs74jl3wxkt3v7yemgfxcbvwyrqusc.py
# Topologically Sorted Source Nodes: [x_111, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_111 => add_187, mul_236, mul_237, sub_65
#   x_112 => relu_62
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_531), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_533), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_236, %unsqueeze_535), kwargs = {})
#   %add_187 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_237, %unsqueeze_537), kwargs = {})
#   %relu_62 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_187,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: inductor_cache/pf/cpfmzzwv5vxlwb7qsa6byji2tfg66e5ljhoympmux3g4zp5ci2ec.py
# Topologically Sorted Source Nodes: [logit1, logit2, logit3, logit4, mul, mul_1, add, mul_2, add_1, mul_3, logit], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add => add_188
#   add_1 => add_189
#   logit => add_190
#   logit1 => convolution_66
#   logit2 => convolution_67
#   logit3 => convolution_68
#   logit4 => convolution_69
#   mul => mul_238
#   mul_1 => mul_239
#   mul_2 => mul_240
#   mul_3 => mul_241
# Graph fragment:
#   %convolution_66 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_35, %primals_332, %primals_333, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_67 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %primals_334, %primals_335, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_68 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_50, %primals_336, %primals_337, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_69 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_62, %primals_338, %primals_339, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %convolution_66), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %convolution_67), kwargs = {})
#   %add_188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %mul_239), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %convolution_68), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_188, %mul_240), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %convolution_69), kwargs = {})
#   %add_190 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_189, %mul_241), kwargs = {})
triton_poi_fused_add_convolution_mul_40 = async_compile.triton('triton_poi_fused_add_convolution_mul_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_40', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_40(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_out_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_out_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr2 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp12 = tl.load(in_out_ptr3 + (x0), None)
    tmp13 = tl.load(in_ptr3 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp16 = tl.load(in_ptr4 + (1))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp19 = tl.load(in_ptr4 + (2))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp23 = tl.load(in_ptr4 + (3))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp27 = tl.load(in_ptr4 + (4))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp7 = tmp4 + tmp6
    tmp11 = tmp8 + tmp10
    tmp15 = tmp12 + tmp14
    tmp18 = tmp17 * tmp3
    tmp21 = tmp20 * tmp7
    tmp22 = tmp18 + tmp21
    tmp25 = tmp24 * tmp11
    tmp26 = tmp22 + tmp25
    tmp29 = tmp28 * tmp15
    tmp30 = tmp26 + tmp29
    tl.store(in_out_ptr0 + (x0), tmp3, None)
    tl.store(in_out_ptr1 + (x0), tmp7, None)
    tl.store(in_out_ptr2 + (x0), tmp11, None)
    tl.store(in_out_ptr3 + (x0), tmp15, None)
    tl.store(out_ptr0 + (x0), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/gj/cgju33gwrkgm6grmktijezypk7256rf4cofklonvkhhwog7yjg6n.py
# Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   logit_1 => convert_element_type_173
# Graph fragment:
#   %convert_element_type_173 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_41 = async_compile.triton('triton_poi_fused__to_copy_41', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_41(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3a/c3aovbvkk2t7m6tujsb2xdeqoj4rar27q63iwioq2josnyxrjbnw.py
# Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   logit_1 => add_192, clamp_max
# Graph fragment:
#   %add_192 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_173, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_192, 31), kwargs = {})
triton_poi_fused_add_clamp_42 = async_compile.triton('triton_poi_fused_add_clamp_42', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_42(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 31, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2c/c2cz77yrrot5llteuqvndnvlqyukcieqjrygzqfmk23ib6gefusu.py
# Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   logit_1 => add_191, clamp_max_2, clamp_min, clamp_min_2, convert_element_type_172, iota_20, mul_242, sub_66, sub_68
# Graph fragment:
#   %iota_20 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_172 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_20, torch.float32), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_172, 0.5), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_191, 0.5), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_242, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_66, 0.0), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_175), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_68, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_43 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_43', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_43(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vo/cvopkle7zlxssd77borzonco2oomng2hixwlyhxp7dwq6qkhg7fs.py
# Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   logit_1 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, add_195, add_196, add_197, mul_244, mul_245, mul_246, sub_69, sub_70, sub_72
# Graph fragment:
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_190, [None, None, %convert_element_type_173, %convert_element_type_175]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_190, [None, None, %convert_element_type_173, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_190, [None, None, %clamp_max, %convert_element_type_175]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_190, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %clamp_max_2), kwargs = {})
#   %add_195 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_244), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %clamp_max_2), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_245), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_196, %add_195), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %clamp_max_3), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_195, %mul_246), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_44 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (tmp8 + 32*tmp22 + 1024*x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (tmp13 + 32*tmp22 + 1024*x2), None, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tl.store(in_out_ptr0 + (x4), tmp31, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340 = args
    args.clear()
    assert_size_stride(primals_1, (4, 1, 64, 64), (4096, 4096, 64, 1))
    assert_size_stride(primals_2, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (128, ), (1, ))
    assert_size_stride(primals_75, (128, ), (1, ))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (128, ), (1, ))
    assert_size_stride(primals_82, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, ), (1, ))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (256, ), (1, ))
    assert_size_stride(primals_121, (256, ), (1, ))
    assert_size_stride(primals_122, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_133, (256, ), (1, ))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_138, (256, ), (1, ))
    assert_size_stride(primals_139, (256, ), (1, ))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (256, ), (1, ))
    assert_size_stride(primals_142, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_143, (256, ), (1, ))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_147, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (512, ), (1, ))
    assert_size_stride(primals_151, (512, ), (1, ))
    assert_size_stride(primals_152, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_153, (512, ), (1, ))
    assert_size_stride(primals_154, (512, ), (1, ))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (512, ), (1, ))
    assert_size_stride(primals_157, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_158, (512, ), (1, ))
    assert_size_stride(primals_159, (512, ), (1, ))
    assert_size_stride(primals_160, (512, ), (1, ))
    assert_size_stride(primals_161, (512, ), (1, ))
    assert_size_stride(primals_162, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_163, (512, ), (1, ))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_168, (512, ), (1, ))
    assert_size_stride(primals_169, (512, ), (1, ))
    assert_size_stride(primals_170, (512, ), (1, ))
    assert_size_stride(primals_171, (512, ), (1, ))
    assert_size_stride(primals_172, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_174, (512, ), (1, ))
    assert_size_stride(primals_175, (512, ), (1, ))
    assert_size_stride(primals_176, (512, ), (1, ))
    assert_size_stride(primals_177, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (512, ), (1, ))
    assert_size_stride(primals_182, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_183, (32, ), (1, ))
    assert_size_stride(primals_184, (32, ), (1, ))
    assert_size_stride(primals_185, (32, ), (1, ))
    assert_size_stride(primals_186, (32, ), (1, ))
    assert_size_stride(primals_187, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_188, (32, ), (1, ))
    assert_size_stride(primals_189, (32, ), (1, ))
    assert_size_stride(primals_190, (32, ), (1, ))
    assert_size_stride(primals_191, (32, ), (1, ))
    assert_size_stride(primals_192, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_193, (64, ), (1, ))
    assert_size_stride(primals_194, (64, ), (1, ))
    assert_size_stride(primals_195, (64, ), (1, ))
    assert_size_stride(primals_196, (64, ), (1, ))
    assert_size_stride(primals_197, (32, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_198, (32, ), (1, ))
    assert_size_stride(primals_199, (32, ), (1, ))
    assert_size_stride(primals_200, (32, ), (1, ))
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_203, (32, ), (1, ))
    assert_size_stride(primals_204, (32, ), (1, ))
    assert_size_stride(primals_205, (32, ), (1, ))
    assert_size_stride(primals_206, (32, ), (1, ))
    assert_size_stride(primals_207, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_208, (64, ), (1, ))
    assert_size_stride(primals_209, (64, ), (1, ))
    assert_size_stride(primals_210, (64, ), (1, ))
    assert_size_stride(primals_211, (64, ), (1, ))
    assert_size_stride(primals_212, (32, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_213, (32, ), (1, ))
    assert_size_stride(primals_214, (32, ), (1, ))
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (32, ), (1, ))
    assert_size_stride(primals_217, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_218, (32, ), (1, ))
    assert_size_stride(primals_219, (32, ), (1, ))
    assert_size_stride(primals_220, (32, ), (1, ))
    assert_size_stride(primals_221, (32, ), (1, ))
    assert_size_stride(primals_222, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, ), (1, ))
    assert_size_stride(primals_225, (64, ), (1, ))
    assert_size_stride(primals_226, (64, ), (1, ))
    assert_size_stride(primals_227, (64, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_228, (64, ), (1, ))
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_230, (64, ), (1, ))
    assert_size_stride(primals_231, (64, ), (1, ))
    assert_size_stride(primals_232, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_233, (64, ), (1, ))
    assert_size_stride(primals_234, (64, ), (1, ))
    assert_size_stride(primals_235, (64, ), (1, ))
    assert_size_stride(primals_236, (64, ), (1, ))
    assert_size_stride(primals_237, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_238, (128, ), (1, ))
    assert_size_stride(primals_239, (128, ), (1, ))
    assert_size_stride(primals_240, (128, ), (1, ))
    assert_size_stride(primals_241, (128, ), (1, ))
    assert_size_stride(primals_242, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_243, (64, ), (1, ))
    assert_size_stride(primals_244, (64, ), (1, ))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (64, ), (1, ))
    assert_size_stride(primals_247, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_248, (64, ), (1, ))
    assert_size_stride(primals_249, (64, ), (1, ))
    assert_size_stride(primals_250, (64, ), (1, ))
    assert_size_stride(primals_251, (64, ), (1, ))
    assert_size_stride(primals_252, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_253, (128, ), (1, ))
    assert_size_stride(primals_254, (128, ), (1, ))
    assert_size_stride(primals_255, (128, ), (1, ))
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (64, ), (1, ))
    assert_size_stride(primals_261, (64, ), (1, ))
    assert_size_stride(primals_262, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_263, (64, ), (1, ))
    assert_size_stride(primals_264, (64, ), (1, ))
    assert_size_stride(primals_265, (64, ), (1, ))
    assert_size_stride(primals_266, (64, ), (1, ))
    assert_size_stride(primals_267, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (128, ), (1, ))
    assert_size_stride(primals_271, (128, ), (1, ))
    assert_size_stride(primals_272, (128, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_273, (128, ), (1, ))
    assert_size_stride(primals_274, (128, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (128, ), (1, ))
    assert_size_stride(primals_277, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_278, (128, ), (1, ))
    assert_size_stride(primals_279, (128, ), (1, ))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, ), (1, ))
    assert_size_stride(primals_287, (128, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_288, (128, ), (1, ))
    assert_size_stride(primals_289, (128, ), (1, ))
    assert_size_stride(primals_290, (128, ), (1, ))
    assert_size_stride(primals_291, (128, ), (1, ))
    assert_size_stride(primals_292, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_293, (128, ), (1, ))
    assert_size_stride(primals_294, (128, ), (1, ))
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_298, (256, ), (1, ))
    assert_size_stride(primals_299, (256, ), (1, ))
    assert_size_stride(primals_300, (256, ), (1, ))
    assert_size_stride(primals_301, (256, ), (1, ))
    assert_size_stride(primals_302, (128, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_303, (128, ), (1, ))
    assert_size_stride(primals_304, (128, ), (1, ))
    assert_size_stride(primals_305, (128, ), (1, ))
    assert_size_stride(primals_306, (128, ), (1, ))
    assert_size_stride(primals_307, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_308, (128, ), (1, ))
    assert_size_stride(primals_309, (128, ), (1, ))
    assert_size_stride(primals_310, (128, ), (1, ))
    assert_size_stride(primals_311, (128, ), (1, ))
    assert_size_stride(primals_312, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_313, (256, ), (1, ))
    assert_size_stride(primals_314, (256, ), (1, ))
    assert_size_stride(primals_315, (256, ), (1, ))
    assert_size_stride(primals_316, (256, ), (1, ))
    assert_size_stride(primals_317, (128, 576, 3, 3), (5184, 9, 3, 1))
    assert_size_stride(primals_318, (128, ), (1, ))
    assert_size_stride(primals_319, (128, ), (1, ))
    assert_size_stride(primals_320, (128, ), (1, ))
    assert_size_stride(primals_321, (128, ), (1, ))
    assert_size_stride(primals_322, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_323, (128, ), (1, ))
    assert_size_stride(primals_324, (128, ), (1, ))
    assert_size_stride(primals_325, (128, ), (1, ))
    assert_size_stride(primals_326, (128, ), (1, ))
    assert_size_stride(primals_327, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_328, (256, ), (1, ))
    assert_size_stride(primals_329, (256, ), (1, ))
    assert_size_stride(primals_330, (256, ), (1, ))
    assert_size_stride(primals_331, (256, ), (1, ))
    assert_size_stride(primals_332, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_333, (1, ), (1, ))
    assert_size_stride(primals_334, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_335, (1, ), (1, ))
    assert_size_stride(primals_336, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_337, (1, ), (1, ))
    assert_size_stride(primals_338, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_339, (1, ), (1, ))
    assert_size_stride(primals_340, (5, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 262144, grid=grid(262144), stream=stream0)
        del primals_6
        buf2 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(buf1, buf2, buf3, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf5 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, primals_8, primals_9, primals_10, primals_11, buf5, 65536, grid=grid(65536), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf7 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf6, primals_13, primals_14, primals_15, primals_16, buf2, buf7, 65536, grid=grid(65536), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf9 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf8, primals_18, primals_19, primals_20, primals_21, buf9, 65536, grid=grid(65536), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf11 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf10, primals_23, primals_24, primals_25, primals_26, buf7, buf11, 65536, grid=grid(65536), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf13 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf12, primals_28, primals_29, primals_30, primals_31, buf13, 65536, grid=grid(65536), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf15 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf107 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf105 = reinterpret_tensor(buf107, (4, 64, 16, 16), (65536, 256, 16, 1), 0)  # alias
        buf139 = empty_strided_cuda((4, 512, 16, 16), (131072, 256, 16, 1), torch.float32)
        buf136 = reinterpret_tensor(buf139, (4, 64, 16, 16), (131072, 256, 16, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [out_18, out_19, out_20, x_48, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_4.run(buf14, primals_33, primals_34, primals_35, primals_36, buf11, buf15, buf105, buf136, 65536, grid=grid(65536), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_37, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf17 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf16, primals_38, primals_39, primals_40, primals_41, buf17, 32768, grid=grid(32768), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 128, 8, 8), (8192, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf15, primals_47, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf20 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [out_25, input_2, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf21, buf18, primals_43, primals_44, primals_45, primals_46, buf19, primals_48, primals_49, primals_50, primals_51, 32768, grid=grid(32768), stream=stream0)
        del primals_46
        del primals_51
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf23 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_29, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf22, primals_53, primals_54, primals_55, primals_56, buf23, 32768, grid=grid(32768), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf25 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_32, out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf24, primals_58, primals_59, primals_60, primals_61, buf21, buf25, 32768, grid=grid(32768), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [out_35], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf27 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_36, out_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf26, primals_63, primals_64, primals_65, primals_66, buf27, 32768, grid=grid(32768), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf29 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_39, out_40, out_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf28, primals_68, primals_69, primals_70, primals_71, buf25, buf29, 32768, grid=grid(32768), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf31 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_43, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf30, primals_73, primals_74, primals_75, primals_76, buf31, 32768, grid=grid(32768), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [out_45], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf33 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        buf130 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        buf128 = reinterpret_tensor(buf130, (4, 128, 8, 8), (32768, 64, 8, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [out_46, out_47, out_48, x_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_8.run(buf32, primals_78, primals_79, primals_80, primals_81, buf29, buf33, buf128, 32768, grid=grid(32768), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [out_49], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_82, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf35 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_50, out_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf34, primals_83, primals_84, primals_85, primals_86, buf35, 16384, grid=grid(16384), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 256, 4, 4), (4096, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf33, primals_92, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf38 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [out_53, input_4, out_54, out_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf39, buf36, primals_88, primals_89, primals_90, primals_91, buf37, primals_93, primals_94, primals_95, primals_96, 16384, grid=grid(16384), stream=stream0)
        del primals_91
        del primals_96
        # Topologically Sorted Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf41 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf40, primals_98, primals_99, primals_100, primals_101, buf41, 16384, grid=grid(16384), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [out_59], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf43 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_60, out_61, out_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf42, primals_103, primals_104, primals_105, primals_106, buf39, buf43, 16384, grid=grid(16384), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [out_63], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf45 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_64, out_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf44, primals_108, primals_109, primals_110, primals_111, buf45, 16384, grid=grid(16384), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [out_66], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf47 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_67, out_68, out_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf46, primals_113, primals_114, primals_115, primals_116, buf43, buf47, 16384, grid=grid(16384), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf49 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf48, primals_118, primals_119, primals_120, primals_121, buf49, 16384, grid=grid(16384), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf51 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_74, out_75, out_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf50, primals_123, primals_124, primals_125, primals_126, buf47, buf51, 16384, grid=grid(16384), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [out_77], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf53 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_78, out_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf52, primals_128, primals_129, primals_130, primals_131, buf53, 16384, grid=grid(16384), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf55 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_81, out_82, out_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf54, primals_133, primals_134, primals_135, primals_136, buf51, buf55, 16384, grid=grid(16384), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [out_84], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf57 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_85, out_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf56, primals_138, primals_139, primals_140, primals_141, buf57, 16384, grid=grid(16384), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [out_87], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf59 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_88, out_89, out_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf58, primals_143, primals_144, primals_145, primals_146, buf55, buf59, 16384, grid=grid(16384), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [out_91], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_147, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf61 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_92, out_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf60, primals_148, primals_149, primals_150, primals_151, buf61, 8192, grid=grid(8192), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [out_94], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 512, 2, 2), (2048, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf59, primals_157, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf64 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [out_95, input_6, out_96, out_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf65, buf62, primals_153, primals_154, primals_155, primals_156, buf63, primals_158, primals_159, primals_160, primals_161, 8192, grid=grid(8192), stream=stream0)
        del primals_156
        del primals_161
        # Topologically Sorted Source Nodes: [out_98], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf67 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_99, out_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf66, primals_163, primals_164, primals_165, primals_166, buf67, 8192, grid=grid(8192), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [out_101], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf69 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_102, out_103, out_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf68, primals_168, primals_169, primals_170, primals_171, buf65, buf69, 8192, grid=grid(8192), stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [out_105], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf71 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_106, out_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf70, primals_173, primals_174, primals_175, primals_176, buf71, 8192, grid=grid(8192), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [out_108], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf73 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        buf169 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_109, out_110, out_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_15.run(buf72, primals_178, primals_179, primals_180, primals_181, buf69, buf73, buf169, 8192, grid=grid(8192), stream=stream0)
        del primals_181
        buf74 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_16.run(buf74, 32, grid=grid(32), stream=stream0)
        buf75 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_17.run(buf1, buf74, buf15, buf75, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf77 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf76, primals_183, primals_184, primals_185, primals_186, buf77, 131072, grid=grid(131072), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf79 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf78, primals_188, primals_189, primals_190, primals_191, buf79, 131072, grid=grid(131072), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf81 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf80, primals_193, primals_194, primals_195, primals_196, buf81, 262144, grid=grid(262144), stream=stream0)
        del primals_196
        buf82 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_19.run(buf82, 16, grid=grid(16), stream=stream0)
        buf83 = empty_strided_cuda((4, 192, 16, 16), (49152, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf15, buf82, buf33, buf83, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf85 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_17, x_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf84, primals_198, primals_199, primals_200, primals_201, buf85, 32768, grid=grid(32768), stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf87 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf86, primals_203, primals_204, primals_205, primals_206, buf87, 32768, grid=grid(32768), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf89 = reinterpret_tensor(buf107, (4, 64, 16, 16), (65536, 256, 16, 1), 16384)  # alias
        buf137 = reinterpret_tensor(buf139, (4, 64, 16, 16), (131072, 256, 16, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [x_23, x_24, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22.run(buf88, primals_208, primals_209, primals_210, primals_211, buf89, buf137, 65536, grid=grid(65536), stream=stream0)
        buf90 = empty_strided_cuda((4, 192, 32, 32), (196608, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf1, buf81, buf74, buf89, buf90, 786432, grid=grid(786432), stream=stream0)
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf92 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf91, primals_213, primals_214, primals_215, primals_216, buf92, 131072, grid=grid(131072), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf94 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf93, primals_218, primals_219, primals_220, primals_221, buf94, 131072, grid=grid(131072), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf96 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_34, x_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf95, primals_223, primals_224, primals_225, primals_226, buf96, 262144, grid=grid(262144), stream=stream0)
        del primals_226
        buf97 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_24.run(buf97, 8, grid=grid(8), stream=stream0)
        buf98 = empty_strided_cuda((4, 384, 8, 8), (24576, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_25.run(buf33, buf97, buf59, buf98, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf100 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf99, primals_228, primals_229, primals_230, primals_231, buf100, 16384, grid=grid(16384), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf102 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_42, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf101, primals_233, primals_234, primals_235, primals_236, buf102, 16384, grid=grid(16384), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf104 = reinterpret_tensor(buf130, (4, 128, 8, 8), (32768, 64, 8, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [x_45, x_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf103, primals_238, primals_239, primals_240, primals_241, buf104, 32768, grid=grid(32768), stream=stream0)
        buf106 = reinterpret_tensor(buf107, (4, 128, 16, 16), (65536, 256, 16, 1), 32768)  # alias
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_28.run(buf82, buf104, buf106, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf109 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_50, x_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf108, primals_243, primals_244, primals_245, primals_246, buf109, 65536, grid=grid(65536), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf111 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_53, x_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf110, primals_248, primals_249, primals_250, primals_251, buf111, 65536, grid=grid(65536), stream=stream0)
        del primals_251
        # Topologically Sorted Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf113 = reinterpret_tensor(buf139, (4, 128, 16, 16), (131072, 256, 16, 1), 32768)  # alias
        # Topologically Sorted Source Nodes: [x_56, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf112, primals_253, primals_254, primals_255, primals_256, buf113, 131072, grid=grid(131072), stream=stream0)
        buf114 = empty_strided_cuda((4, 320, 32, 32), (327680, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_30.run(buf1, buf81, buf96, buf74, buf113, buf114, 1310720, grid=grid(1310720), stream=stream0)
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf116 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_61, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf115, primals_258, primals_259, primals_260, primals_261, buf116, 262144, grid=grid(262144), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf118 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_64, x_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf117, primals_263, primals_264, primals_265, primals_266, buf118, 262144, grid=grid(262144), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf120 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf119, primals_268, primals_269, primals_270, primals_271, buf120, 524288, grid=grid(524288), stream=stream0)
        del primals_271
        buf121 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_32.run(buf121, 4, grid=grid(4), stream=stream0)
        buf122 = empty_strided_cuda((4, 768, 4, 4), (12288, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_33.run(buf59, buf121, buf73, buf122, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_272, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf124 = reinterpret_tensor(buf73, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_72, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf123, primals_273, primals_274, primals_275, primals_276, buf124, 8192, grid=grid(8192), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf126 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_75, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf125, primals_278, primals_279, primals_280, primals_281, buf126, 8192, grid=grid(8192), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf129 = reinterpret_tensor(buf130, (4, 256, 8, 8), (32768, 64, 8, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [x_78, x_79, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_35.run(buf97, buf127, primals_283, primals_284, primals_285, primals_286, buf129, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf132 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_83, x_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf131, primals_288, primals_289, primals_290, primals_291, buf132, 32768, grid=grid(32768), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [x_85], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf134 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_86, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf133, primals_293, primals_294, primals_295, primals_296, buf134, 32768, grid=grid(32768), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf138 = reinterpret_tensor(buf139, (4, 256, 16, 16), (131072, 256, 16, 1), 65536)  # alias
        # Topologically Sorted Source Nodes: [x_89, x_90, x_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_36.run(buf82, buf135, primals_298, primals_299, primals_300, primals_301, buf138, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf141 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_94, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf140, primals_303, primals_304, primals_305, primals_306, buf141, 131072, grid=grid(131072), stream=stream0)
        del primals_306
        # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf143 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_97, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf142, primals_308, primals_309, primals_310, primals_311, buf143, 131072, grid=grid(131072), stream=stream0)
        del primals_311
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf145 = empty_strided_cuda((4, 576, 32, 32), (589824, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_103], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf1, buf81, buf96, buf120, buf74, buf144, primals_313, primals_314, primals_315, primals_316, buf145, 2359296, grid=grid(2359296), stream=stream0)
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf147 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_105, x_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf146, primals_318, primals_319, primals_320, primals_321, buf147, 524288, grid=grid(524288), stream=stream0)
        del primals_321
        # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf149 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_108, x_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf148, primals_323, primals_324, primals_325, primals_326, buf149, 524288, grid=grid(524288), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 256, 32, 32), (262144, 1024, 32, 1))
        buf151 = empty_strided_cuda((4, 256, 32, 32), (262144, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_111, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf150, primals_328, primals_329, primals_330, primals_331, buf151, 1048576, grid=grid(1048576), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [logit1], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf81, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 1, 32, 32), (1024, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [logit2], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf96, primals_334, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 1, 32, 32), (1024, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [logit3], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf120, primals_336, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 1, 32, 32), (1024, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [logit4], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf151, primals_338, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 1, 32, 32), (1024, 1024, 32, 1))
        buf153 = buf152; del buf152  # reuse
        buf155 = buf154; del buf154  # reuse
        buf157 = buf156; del buf156  # reuse
        buf159 = buf158; del buf158  # reuse
        buf160 = empty_strided_cuda((4, 1, 32, 32), (1024, 4096, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logit1, logit2, logit3, logit4, mul, mul_1, add, mul_2, add_1, mul_3, logit], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_40.run(buf153, buf155, buf157, buf159, primals_333, primals_335, primals_337, primals_339, primals_340, buf160, 4096, grid=grid(4096), stream=stream0)
        del primals_333
        del primals_335
        del primals_337
        del primals_339
        buf161 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_41.run(buf161, 64, grid=grid(64), stream=stream0)
        buf162 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_42.run(buf162, 64, grid=grid(64), stream=stream0)
        buf163 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_41.run(buf163, 64, grid=grid(64), stream=stream0)
        buf164 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_42.run(buf164, 64, grid=grid(64), stream=stream0)
        buf165 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_43.run(buf165, 64, grid=grid(64), stream=stream0)
        buf167 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_43.run(buf167, 64, grid=grid(64), stream=stream0)
        buf166 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf168 = reinterpret_tensor(buf166, (4, 1, 64, 64), (4096, 4096, 64, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [logit_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_44.run(buf168, buf161, buf163, buf160, buf164, buf165, buf162, buf167, 16384, grid=grid(16384), stream=stream0)
        del buf160
    return (buf168, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_268, primals_269, primals_270, primals_272, primals_273, primals_274, primals_275, primals_277, primals_278, primals_279, primals_280, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_307, primals_308, primals_309, primals_310, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_334, primals_336, primals_338, primals_340, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf107, buf108, buf109, buf110, buf111, buf112, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf130, buf131, buf132, buf133, buf134, buf135, buf139, buf140, buf141, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf153, buf155, buf157, buf159, buf161, buf162, buf163, buf164, buf165, buf167, buf169, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 1, 64, 64), (4096, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((32, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((32, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((64, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((128, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((128, 576, 3, 3), (5184, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((5, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
