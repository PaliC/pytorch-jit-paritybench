# AOT ID: ['107_forward']
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


# kernel path: inductor_cache/kz/ckzav56m6x4e5ozwrluwzajfeuvjyj7dznixacmxfmyrghr4orz6.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ko/ckoidgcxmfqmhvevmbelljzdwj3e3nm2osp7fw7hcc3wlqsfqj26.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_3 => getitem, getitem_1
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


# kernel path: inductor_cache/fq/cfqdwgebzfrnz43uspghxsaiikgzafzko6esyto2lhivqd23lv7g.py
# Topologically Sorted Source Nodes: [conv2d_1, batch_norm_1, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_4, mul_5, sub_1
#   conv2d_1 => convolution_1
#   out => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/m4/cm4xulhhm65mbkwx6hpoptwh2lkntiltcc6h5n5loqsmd4dsw472.py
# Topologically Sorted Source Nodes: [conv2d_3, out_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv2d_3 => convolution_3
#   out_2 => add_7, mul_10, mul_11, sub_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_20, %primals_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/vi/cviafgsxydzp64jzxod47l7namzcatxegsdeqpnyhp5rcnpkwvtj.py
# Topologically Sorted Source Nodes: [context_mask_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   context_mask_2 => amax, div, exp, sub_4, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_1, [2], True), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [2], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_per_fused__softmax_4 = async_compile.triton('triton_per_fused__softmax_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_4(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 4
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 256*x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp4, 0))
    tmp7 = tmp3 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp8 / tmp11
    tl.store(in_out_ptr0 + (r1 + 256*x0), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/r7/cr7nkw62la2fozwubtpnlpqisdlrhc6bhgril24kbiyjl7ut47ct.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution_5
#   input_2 => add_8, add_9, mul_12, mul_13, rsqrt, sub_5, var_mean
#   input_3 => relu_3
# Graph fragment:
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_5, %primals_28, %primals_29, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_5, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %getitem_3), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %primals_30), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %primals_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_per_fused_convolution_native_layer_norm_relu_5 = async_compile.triton('triton_per_fused_convolution_native_layer_norm_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_layer_norm_relu_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_layer_norm_relu_5(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 128.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp2 - tmp12
    tmp25 = tmp24 * tmp23
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1, 1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, xmask)
    tl.store(out_ptr1 + (r1 + 128*x0), tmp31, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j7/cj7hpjdbbdiky2c3v4wtab5mw5znqcfjhjh6clywissabznxdlw7.py
# Topologically Sorted Source Nodes: [input_4, out_3, out_4, input_5], Original ATen: [aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_4 => convolution_6
#   input_5 => relu_4
#   out_3 => add_10
#   out_4 => add_11
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %primals_32, %primals_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %convolution_6), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %getitem), kwargs = {})
#   %relu_4 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused_add_convolution_relu_6 = async_compile.triton('triton_poi_fused_add_convolution_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 256
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/oe/coe67zgjrrgtjtg6d3abt65gdljpulc2k3k2y4gdho6hxkbqrpnc.py
# Topologically Sorted Source Nodes: [conv2d_19, batch_norm_10, out_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_10 => add_33, mul_37, mul_38, sub_16
#   conv2d_19 => convolution_19
#   out_15 => relu_13
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %primals_86, %primals_87, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_87), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_89), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_91), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_93), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/kx/ckxphjs7ezoxlz72fteiabpwkn6jrjtmvp73jsflul4ff2msgp3i.py
# Topologically Sorted Source Nodes: [conv2d_20, batch_norm_11, out_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_11 => add_35, mul_40, mul_41, sub_17
#   conv2d_20 => convolution_20
#   out_16 => relu_14
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_92, %primals_93, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_95), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_97), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_99), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_101), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_35,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/vg/cvgesvdqo5cn6vsryblubxx7gj5q3vib3nxfjb62zmtxwyhli7bz.py
# Topologically Sorted Source Nodes: [conv2d_21, out_17], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv2d_21 => convolution_21
#   out_17 => add_37, mul_43, mul_44, sub_18
# Graph fragment:
#   %convolution_21 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %primals_98, %primals_99, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_103), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_105), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_107), kwargs = {})
#   %add_37 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_109), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/b7/cb7ll5lprjk4uzj3jhxkbtb3uxlwrdyp4zkvv3owuleuxcclch3r.py
# Topologically Sorted Source Nodes: [context_mask_14], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   context_mask_14 => amax_3, div_3, exp_3, sub_19, sum_4
# Graph fragment:
#   %amax_3 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_19, [2], True), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_19, %amax_3), kwargs = {})
#   %exp_3 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_19,), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_3, [2], True), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_3, %sum_4), kwargs = {})
triton_per_fused__softmax_10 = async_compile.triton('triton_per_fused__softmax_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_10(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tmp9 / tmp13
    tl.store(in_out_ptr0 + (r1 + 64*x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ff/cff43c5igon2rezikmtqxlgulhitd3cmossfb2b754b6lfxni3j4.py
# Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
# Source node to ATen node mapping:
#   input_16 => convolution_23
#   input_17 => add_38, add_39, mul_45, mul_46, rsqrt_3, sub_20, var_mean_3
#   input_18 => relu_15
# Graph fragment:
#   %convolution_23 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_23, %primals_106, %primals_107, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_23, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_38,), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %getitem_9), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %rsqrt_3), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %primals_108), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %primals_109), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_39,), kwargs = {})
triton_per_fused_convolution_native_layer_norm_relu_11 = async_compile.triton('triton_per_fused_convolution_native_layer_norm_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_layer_norm_relu_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_layer_norm_relu_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 256*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 256.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp2 - tmp10
    tmp22 = tmp21 * tmp20
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full([1], 0, tl.int32)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tl.store(in_out_ptr0 + (r1 + 256*x0), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp20, None)
    tl.store(out_ptr1 + (r1 + 256*x0), tmp28, None)
    tl.store(out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/5a/c5akwa6yvxrrebbd7wrmxujam62cggnyq4kv64wrxc4r5pzeny6e.py
# Topologically Sorted Source Nodes: [input_19, out_18, input_20, input_21, out_19, input_22], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_19 => convolution_24
#   input_20 => convolution_25
#   input_21 => add_42, mul_48, mul_49, sub_21
#   input_22 => relu_16
#   out_18 => add_40
#   out_19 => add_43
# Graph fragment:
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_110, %primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %convolution_24), kwargs = {})
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %primals_112, %primals_113, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_113), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_115), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %unsqueeze_117), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_119), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %add_42), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_43,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    x4 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp2 - tmp8
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp14 / tmp13
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tmp7 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/hf/chfzuk46uea4uikfbsl5exq5pai2ef5h5fjeymqajb26jpxmoqfx.py
# Topologically Sorted Source Nodes: [input_26, out_23, out_24, input_27], Original ATen: [aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_26 => convolution_31
#   input_27 => relu_20
#   out_23 => add_52
#   out_24 => add_53
# Graph fragment:
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %primals_142, %primals_143, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %convolution_31), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_52, %relu_16), kwargs = {})
#   %relu_20 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
triton_poi_fused_add_convolution_relu_13 = async_compile.triton('triton_poi_fused_add_convolution_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 64
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/ao/caobyfd72tmymp5rrlew3f34nbeb7zrbomgzn3gz4h5elqid2l26.py
# Topologically Sorted Source Nodes: [conv2d_44, batch_norm_23, out_35], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_23 => add_75, mul_84, mul_85, sub_37
#   conv2d_44 => convolution_44
#   out_35 => relu_29
# Graph fragment:
#   %convolution_44 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_28, %primals_196, %primals_197, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_199), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_201), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %unsqueeze_203), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %unsqueeze_205), kwargs = {})
#   %relu_29 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_75,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ay/cayio73gqc7csknrplr33abzr4vz3aohiq6h7gylosfinu7iwjhe.py
# Topologically Sorted Source Nodes: [conv2d_45, batch_norm_24, out_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_24 => add_77, mul_87, mul_88, sub_38
#   conv2d_45 => convolution_45
#   out_36 => relu_30
# Graph fragment:
#   %convolution_45 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_29, %primals_202, %primals_203, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_207), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_209), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %unsqueeze_211), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %unsqueeze_213), kwargs = {})
#   %relu_30 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_77,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/nr/cnryebpfhyb7nnlzplhtk2br64i6yxnxcz6ytgondhywxcm6b64m.py
# Topologically Sorted Source Nodes: [conv2d_46, out_37], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv2d_46 => convolution_46
#   out_37 => add_79, mul_90, mul_91, sub_39
# Graph fragment:
#   %convolution_46 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_30, %primals_208, %primals_209, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_215), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_217), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %unsqueeze_219), kwargs = {})
#   %add_79 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %unsqueeze_221), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/w3/cw3lll5hxxykcrglida3njmmbxz6mlq3rxldze4mjeehf55skjde.py
# Topologically Sorted Source Nodes: [context_mask_30], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   context_mask_30 => amax_7, div_7, exp_7, sub_40, sum_8
# Graph fragment:
#   %amax_7 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_43, [2], True), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_43, %amax_7), kwargs = {})
#   %exp_7 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_40,), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_7, [2], True), kwargs = {})
#   %div_7 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_7, %sum_8), kwargs = {})
triton_per_fused__softmax_17 = async_compile.triton('triton_per_fused__softmax_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_17(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tmp9 / tmp13
    tl.store(in_out_ptr0 + (r1 + 16*x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ys/cysjlejkkl7pxop6e5cqvsekmmq6anv42dle2yup7hxlwhttxxcv.py
# Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
# Source node to ATen node mapping:
#   input_38 => convolution_48
#   input_39 => add_80, add_81, mul_92, mul_93, rsqrt_7, sub_41, var_mean_7
#   input_40 => relu_31
# Graph fragment:
#   %convolution_48 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_47, %primals_216, %primals_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_48, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_80,), kwargs = {})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_48, %getitem_17), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %rsqrt_7), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_92, %primals_218), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %primals_219), kwargs = {})
#   %relu_31 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_81,), kwargs = {})
triton_per_fused_convolution_native_layer_norm_relu_18 = async_compile.triton('triton_per_fused_convolution_native_layer_norm_relu_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_layer_norm_relu_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_layer_norm_relu_18(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 512.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp2 - tmp10
    tmp22 = tmp21 * tmp20
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full([1], 0, tl.int32)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tl.store(in_out_ptr0 + (r1 + 512*x0), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp20, None)
    tl.store(out_ptr1 + (r1 + 512*x0), tmp28, None)
    tl.store(out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/mb/cmb77r5gmdujjdlijn6czykbb6s2sjj3jndzolye4yjrmfaoujun.py
# Topologically Sorted Source Nodes: [input_41, out_38, input_42, input_43, out_39, input_44], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_41 => convolution_49
#   input_42 => convolution_50
#   input_43 => add_84, mul_95, mul_96, sub_42
#   input_44 => relu_32
#   out_38 => add_82
#   out_39 => add_85
# Graph fragment:
#   %convolution_49 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_31, %primals_220, %primals_221, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_79, %convolution_49), kwargs = {})
#   %convolution_50 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_28, %primals_222, %primals_223, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_225), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_227), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_95, %unsqueeze_229), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %unsqueeze_231), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_82, %add_84), kwargs = {})
#   %relu_32 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_85,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
    x4 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp2 - tmp8
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp14 / tmp13
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tmp7 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/rs/crs7w7btked7oeqwaqdwerqbqt5xb54ttfzj56wajmustke2chid.py
# Topologically Sorted Source Nodes: [input_48, out_43, out_44, input_49], Original ATen: [aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_48 => convolution_56
#   input_49 => relu_36
#   out_43 => add_94
#   out_44 => add_95
# Graph fragment:
#   %convolution_56 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_35, %primals_252, %primals_253, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %convolution_56), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %relu_32), kwargs = {})
#   %relu_36 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_95,), kwargs = {})
triton_poi_fused_add_convolution_relu_20 = async_compile.triton('triton_poi_fused_add_convolution_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 16
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/g6/cg6g4erc25n3voaxwghkdgieqyhn4cw4l4il2xu7mvzkyipu273p.py
# Topologically Sorted Source Nodes: [conv2d_81, batch_norm_42, out_65], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_42 => add_137, mul_153, mul_154, sub_68
#   conv2d_81 => convolution_81
#   out_65 => relu_53
# Graph fragment:
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_52, %primals_358, %primals_359, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_81, %unsqueeze_363), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_365), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %unsqueeze_367), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_154, %unsqueeze_369), kwargs = {})
#   %relu_53 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_137,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/jj/cjjdnib433jeqbzvtfzook6cpkig23kckr5org6k5nt2gb6cntmh.py
# Topologically Sorted Source Nodes: [conv2d_83, out_67], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv2d_83 => convolution_83
#   out_67 => add_141, mul_159, mul_160, sub_70
# Graph fragment:
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_54, %primals_370, %primals_371, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_379), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_381), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %unsqueeze_383), kwargs = {})
#   %add_141 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_385), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/bd/cbdldxq2unnp3we6iuk4kcfhnbk6adi74pymbibyltfwuce5elqg.py
# Topologically Sorted Source Nodes: [input_70, input_71, input_72], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
# Source node to ATen node mapping:
#   input_70 => convolution_85
#   input_71 => add_142, add_143, mul_161, mul_162, rsqrt_13, sub_72, var_mean_13
#   input_72 => relu_55
# Graph fragment:
#   %convolution_85 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_83, %primals_378, %primals_379, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_85, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_142 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-05), kwargs = {})
#   %rsqrt_13 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_142,), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %getitem_29), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %rsqrt_13), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_161, %primals_380), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_162, %primals_381), kwargs = {})
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_143,), kwargs = {})
triton_per_fused_convolution_native_layer_norm_relu_23 = async_compile.triton('triton_per_fused_convolution_native_layer_norm_relu_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_layer_norm_relu_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_layer_norm_relu_23(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 1024*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 1024.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp2 - tmp10
    tmp22 = tmp21 * tmp20
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full([1], 0, tl.int32)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tl.store(in_out_ptr0 + (r1 + 1024*x0), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp20, None)
    tl.store(out_ptr1 + (r1 + 1024*x0), tmp28, None)
    tl.store(out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/qb/cqbhtxsn4eunurkvn7yu2gi3al6ars5e5qis2bz4uzzfj52vvhyr.py
# Topologically Sorted Source Nodes: [input_73, out_68, input_74, input_75, out_69, input_76], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_73 => convolution_86
#   input_74 => convolution_87
#   input_75 => add_146, mul_164, mul_165, sub_73
#   input_76 => relu_56
#   out_68 => add_144
#   out_69 => add_147
# Graph fragment:
#   %convolution_86 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_55, %primals_382, %primals_383, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_141, %convolution_86), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_52, %primals_384, %primals_385, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_389), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_391), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_164, %unsqueeze_393), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_165, %unsqueeze_395), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_144, %add_146), kwargs = {})
#   %relu_56 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_147,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 512)
    x4 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp2 - tmp8
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp14 / tmp13
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tmp7 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/di/cdi2rxw2t3bdiils6k3vhm2wqb6tbnw6ny5n2cieam6t35x22ejb.py
# Topologically Sorted Source Nodes: [input_80, out_73, out_74, input_81], Original ATen: [aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_80 => convolution_93
#   input_81 => relu_60
#   out_73 => add_156
#   out_74 => add_157
# Graph fragment:
#   %convolution_93 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_59, %primals_414, %primals_415, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_153, %convolution_93), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_156, %relu_56), kwargs = {})
#   %relu_60 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_157,), kwargs = {})
triton_poi_fused_add_convolution_relu_25 = async_compile.triton('triton_poi_fused_add_convolution_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 16
    x1 = ((xindex // 16) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/gj/cgjwhr7xvtgroh7ps6g5xvqzbras3esnqx4o7xw7rwho2cmutg6i.py
# Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_87 => convolution_100
# Graph fragment:
#   %convolution_100 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_64, %primals_442, %primals_443, [2, 2], [1, 1], [1, 1], True, [0, 0], 128), kwargs = {})
triton_poi_fused_convolution_26 = async_compile.triton('triton_poi_fused_convolution_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_26(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/qx/cqx2p7yhfmni6wfrzntshlkjkrrqtr5p3qx5foqt7xqqflonhnif.py
# Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_91 => convolution_102
# Graph fragment:
#   %convolution_102 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_65, %primals_450, %primals_451, [2, 2], [1, 1], [1, 1], True, [0, 0], 64), kwargs = {})
triton_poi_fused_convolution_27 = async_compile.triton('triton_poi_fused_convolution_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_27(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/qs/cqsydw5rsmcvseqzm5za25fpguxwfza5bfib2iwpxyh3yrmb5hpv.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_4 => convolution_104
# Graph fragment:
#   %convolution_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_66, %primals_458, %primals_459, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_28 = async_compile.triton('triton_poi_fused_convolution_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_28(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_27, (1, ), (1, ))
    assert_size_stride(primals_28, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_31, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_32, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, ), (1, ))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_53, (1, ), (1, ))
    assert_size_stride(primals_54, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_57, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_58, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_61, (64, ), (1, ))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_79, (1, ), (1, ))
    assert_size_stride(primals_80, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_81, (128, ), (1, ))
    assert_size_stride(primals_82, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_83, (128, 1, 1), (1, 1, 1))
    assert_size_stride(primals_84, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_92, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_98, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (128, ), (1, ))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_105, (1, ), (1, ))
    assert_size_stride(primals_106, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_109, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_110, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (128, ), (1, ))
    assert_size_stride(primals_118, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_137, (1, ), (1, ))
    assert_size_stride(primals_138, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_139, (256, ), (1, ))
    assert_size_stride(primals_140, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_141, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_142, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (128, ), (1, ))
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (128, ), (1, ))
    assert_size_stride(primals_150, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_163, (1, ), (1, ))
    assert_size_stride(primals_164, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_165, (256, ), (1, ))
    assert_size_stride(primals_166, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_167, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_168, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_171, (128, ), (1, ))
    assert_size_stride(primals_172, (128, ), (1, ))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (128, ), (1, ))
    assert_size_stride(primals_176, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_188, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_189, (1, ), (1, ))
    assert_size_stride(primals_190, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_191, (256, ), (1, ))
    assert_size_stride(primals_192, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_193, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_194, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_197, (256, ), (1, ))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, ), (1, ))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (256, ), (1, ))
    assert_size_stride(primals_202, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_203, (256, ), (1, ))
    assert_size_stride(primals_204, (256, ), (1, ))
    assert_size_stride(primals_205, (256, ), (1, ))
    assert_size_stride(primals_206, (256, ), (1, ))
    assert_size_stride(primals_207, (256, ), (1, ))
    assert_size_stride(primals_208, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_209, (256, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_215, (1, ), (1, ))
    assert_size_stride(primals_216, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_217, (512, ), (1, ))
    assert_size_stride(primals_218, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_219, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_220, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_221, (256, ), (1, ))
    assert_size_stride(primals_222, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_223, (256, ), (1, ))
    assert_size_stride(primals_224, (256, ), (1, ))
    assert_size_stride(primals_225, (256, ), (1, ))
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_227, (256, ), (1, ))
    assert_size_stride(primals_228, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_231, (256, ), (1, ))
    assert_size_stride(primals_232, (256, ), (1, ))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (256, ), (1, ))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_240, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_241, (256, ), (1, ))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_245, (256, ), (1, ))
    assert_size_stride(primals_246, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_247, (1, ), (1, ))
    assert_size_stride(primals_248, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_249, (512, ), (1, ))
    assert_size_stride(primals_250, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_251, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_252, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_254, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_255, (256, ), (1, ))
    assert_size_stride(primals_256, (256, ), (1, ))
    assert_size_stride(primals_257, (256, ), (1, ))
    assert_size_stride(primals_258, (256, ), (1, ))
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_261, (256, ), (1, ))
    assert_size_stride(primals_262, (256, ), (1, ))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (256, ), (1, ))
    assert_size_stride(primals_266, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_267, (256, ), (1, ))
    assert_size_stride(primals_268, (256, ), (1, ))
    assert_size_stride(primals_269, (256, ), (1, ))
    assert_size_stride(primals_270, (256, ), (1, ))
    assert_size_stride(primals_271, (256, ), (1, ))
    assert_size_stride(primals_272, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_273, (1, ), (1, ))
    assert_size_stride(primals_274, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_275, (512, ), (1, ))
    assert_size_stride(primals_276, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_277, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_278, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_279, (256, ), (1, ))
    assert_size_stride(primals_280, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_281, (256, ), (1, ))
    assert_size_stride(primals_282, (256, ), (1, ))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_287, (256, ), (1, ))
    assert_size_stride(primals_288, (256, ), (1, ))
    assert_size_stride(primals_289, (256, ), (1, ))
    assert_size_stride(primals_290, (256, ), (1, ))
    assert_size_stride(primals_291, (256, ), (1, ))
    assert_size_stride(primals_292, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_293, (256, ), (1, ))
    assert_size_stride(primals_294, (256, ), (1, ))
    assert_size_stride(primals_295, (256, ), (1, ))
    assert_size_stride(primals_296, (256, ), (1, ))
    assert_size_stride(primals_297, (256, ), (1, ))
    assert_size_stride(primals_298, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_299, (1, ), (1, ))
    assert_size_stride(primals_300, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_301, (512, ), (1, ))
    assert_size_stride(primals_302, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_303, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_304, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_305, (256, ), (1, ))
    assert_size_stride(primals_306, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_307, (256, ), (1, ))
    assert_size_stride(primals_308, (256, ), (1, ))
    assert_size_stride(primals_309, (256, ), (1, ))
    assert_size_stride(primals_310, (256, ), (1, ))
    assert_size_stride(primals_311, (256, ), (1, ))
    assert_size_stride(primals_312, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_313, (256, ), (1, ))
    assert_size_stride(primals_314, (256, ), (1, ))
    assert_size_stride(primals_315, (256, ), (1, ))
    assert_size_stride(primals_316, (256, ), (1, ))
    assert_size_stride(primals_317, (256, ), (1, ))
    assert_size_stride(primals_318, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_319, (256, ), (1, ))
    assert_size_stride(primals_320, (256, ), (1, ))
    assert_size_stride(primals_321, (256, ), (1, ))
    assert_size_stride(primals_322, (256, ), (1, ))
    assert_size_stride(primals_323, (256, ), (1, ))
    assert_size_stride(primals_324, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_325, (1, ), (1, ))
    assert_size_stride(primals_326, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_327, (512, ), (1, ))
    assert_size_stride(primals_328, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_329, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_330, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_331, (256, ), (1, ))
    assert_size_stride(primals_332, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_333, (256, ), (1, ))
    assert_size_stride(primals_334, (256, ), (1, ))
    assert_size_stride(primals_335, (256, ), (1, ))
    assert_size_stride(primals_336, (256, ), (1, ))
    assert_size_stride(primals_337, (256, ), (1, ))
    assert_size_stride(primals_338, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_339, (256, ), (1, ))
    assert_size_stride(primals_340, (256, ), (1, ))
    assert_size_stride(primals_341, (256, ), (1, ))
    assert_size_stride(primals_342, (256, ), (1, ))
    assert_size_stride(primals_343, (256, ), (1, ))
    assert_size_stride(primals_344, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_345, (256, ), (1, ))
    assert_size_stride(primals_346, (256, ), (1, ))
    assert_size_stride(primals_347, (256, ), (1, ))
    assert_size_stride(primals_348, (256, ), (1, ))
    assert_size_stride(primals_349, (256, ), (1, ))
    assert_size_stride(primals_350, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_351, (1, ), (1, ))
    assert_size_stride(primals_352, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_353, (512, ), (1, ))
    assert_size_stride(primals_354, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_355, (512, 1, 1), (1, 1, 1))
    assert_size_stride(primals_356, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_357, (256, ), (1, ))
    assert_size_stride(primals_358, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_359, (512, ), (1, ))
    assert_size_stride(primals_360, (512, ), (1, ))
    assert_size_stride(primals_361, (512, ), (1, ))
    assert_size_stride(primals_362, (512, ), (1, ))
    assert_size_stride(primals_363, (512, ), (1, ))
    assert_size_stride(primals_364, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_365, (512, ), (1, ))
    assert_size_stride(primals_366, (512, ), (1, ))
    assert_size_stride(primals_367, (512, ), (1, ))
    assert_size_stride(primals_368, (512, ), (1, ))
    assert_size_stride(primals_369, (512, ), (1, ))
    assert_size_stride(primals_370, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_371, (512, ), (1, ))
    assert_size_stride(primals_372, (512, ), (1, ))
    assert_size_stride(primals_373, (512, ), (1, ))
    assert_size_stride(primals_374, (512, ), (1, ))
    assert_size_stride(primals_375, (512, ), (1, ))
    assert_size_stride(primals_376, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_377, (1, ), (1, ))
    assert_size_stride(primals_378, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_379, (1024, ), (1, ))
    assert_size_stride(primals_380, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_381, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_382, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_383, (512, ), (1, ))
    assert_size_stride(primals_384, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_385, (512, ), (1, ))
    assert_size_stride(primals_386, (512, ), (1, ))
    assert_size_stride(primals_387, (512, ), (1, ))
    assert_size_stride(primals_388, (512, ), (1, ))
    assert_size_stride(primals_389, (512, ), (1, ))
    assert_size_stride(primals_390, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_391, (512, ), (1, ))
    assert_size_stride(primals_392, (512, ), (1, ))
    assert_size_stride(primals_393, (512, ), (1, ))
    assert_size_stride(primals_394, (512, ), (1, ))
    assert_size_stride(primals_395, (512, ), (1, ))
    assert_size_stride(primals_396, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_397, (512, ), (1, ))
    assert_size_stride(primals_398, (512, ), (1, ))
    assert_size_stride(primals_399, (512, ), (1, ))
    assert_size_stride(primals_400, (512, ), (1, ))
    assert_size_stride(primals_401, (512, ), (1, ))
    assert_size_stride(primals_402, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_403, (512, ), (1, ))
    assert_size_stride(primals_404, (512, ), (1, ))
    assert_size_stride(primals_405, (512, ), (1, ))
    assert_size_stride(primals_406, (512, ), (1, ))
    assert_size_stride(primals_407, (512, ), (1, ))
    assert_size_stride(primals_408, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_409, (1, ), (1, ))
    assert_size_stride(primals_410, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_411, (1024, ), (1, ))
    assert_size_stride(primals_412, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_413, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_414, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_415, (512, ), (1, ))
    assert_size_stride(primals_416, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_417, (512, ), (1, ))
    assert_size_stride(primals_418, (512, ), (1, ))
    assert_size_stride(primals_419, (512, ), (1, ))
    assert_size_stride(primals_420, (512, ), (1, ))
    assert_size_stride(primals_421, (512, ), (1, ))
    assert_size_stride(primals_422, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_423, (512, ), (1, ))
    assert_size_stride(primals_424, (512, ), (1, ))
    assert_size_stride(primals_425, (512, ), (1, ))
    assert_size_stride(primals_426, (512, ), (1, ))
    assert_size_stride(primals_427, (512, ), (1, ))
    assert_size_stride(primals_428, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_429, (512, ), (1, ))
    assert_size_stride(primals_430, (512, ), (1, ))
    assert_size_stride(primals_431, (512, ), (1, ))
    assert_size_stride(primals_432, (512, ), (1, ))
    assert_size_stride(primals_433, (512, ), (1, ))
    assert_size_stride(primals_434, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_435, (1, ), (1, ))
    assert_size_stride(primals_436, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_437, (1024, ), (1, ))
    assert_size_stride(primals_438, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_439, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_440, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_441, (512, ), (1, ))
    assert_size_stride(primals_442, (512, 2, 4, 4), (32, 16, 4, 1))
    assert_size_stride(primals_443, (256, ), (1, ))
    assert_size_stride(primals_444, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_445, (128, ), (1, ))
    assert_size_stride(primals_446, (128, ), (1, ))
    assert_size_stride(primals_447, (128, ), (1, ))
    assert_size_stride(primals_448, (128, ), (1, ))
    assert_size_stride(primals_449, (128, ), (1, ))
    assert_size_stride(primals_450, (128, 2, 4, 4), (32, 16, 4, 1))
    assert_size_stride(primals_451, (128, ), (1, ))
    assert_size_stride(primals_452, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_453, (64, ), (1, ))
    assert_size_stride(primals_454, (64, ), (1, ))
    assert_size_stride(primals_455, (64, ), (1, ))
    assert_size_stride(primals_456, (64, ), (1, ))
    assert_size_stride(primals_457, (64, ), (1, ))
    assert_size_stride(primals_458, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_459, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, primals_2, primals_4, primals_5, primals_6, primals_7, buf2, 262144, grid=grid(262144), stream=stream0)
        del primals_2
        del primals_7
        buf3 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(buf2, buf3, buf4, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf6 = buf5; del buf5  # reuse
        buf7 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_1, batch_norm_1, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf6, primals_9, primals_10, primals_11, primals_12, primals_13, buf7, 65536, grid=grid(65536), stream=stream0)
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_2, batch_norm_2, out_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf9, primals_15, primals_16, primals_17, primals_18, primals_19, buf10, 65536, grid=grid(65536), stream=stream0)
        del primals_15
        del primals_19
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf12 = buf11; del buf11  # reuse
        buf13 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_3, out_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_3.run(buf12, primals_21, primals_22, primals_23, primals_24, primals_25, buf13, 65536, grid=grid(65536), stream=stream0)
        del primals_21
        del primals_25
        # Topologically Sorted Source Nodes: [context_mask], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 1, 16, 16), (256, 256, 16, 1))
        buf17 = reinterpret_tensor(buf14, (4, 1, 256), (256, 256, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [context_mask_2], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_4.run(buf17, primals_27, 4, 256, grid=grid(4), stream=stream0)
        del primals_27
        buf18 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (4, 64, 256), (16384, 256, 1), 0), reinterpret_tensor(buf17, (4, 256, 1), (256, 1, 1), 0), out=buf18)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(reinterpret_tensor(buf18, (4, 64, 1, 1), (64, 1, 1, 1), 0), primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 128, 1, 1), (128, 1, 1, 1))
        buf20 = buf19; del buf19  # reuse
        buf21 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf22 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf24 = reinterpret_tensor(buf22, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf22  # reuse
        buf25 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_5.run(buf20, buf24, primals_29, primals_30, primals_31, buf21, buf25, 4, 128, grid=grid(4), stream=stream0)
        del primals_29
        del primals_31
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 64, 1, 1), (64, 1, 1, 1))
        buf27 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, out_3, out_4, input_5], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_6.run(buf13, buf26, primals_33, buf3, buf27, 65536, grid=grid(65536), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf29 = buf28; del buf28  # reuse
        buf30 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_7, batch_norm_4, out_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf29, primals_35, primals_36, primals_37, primals_38, primals_39, buf30, 65536, grid=grid(65536), stream=stream0)
        del primals_35
        del primals_39
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf32 = buf31; del buf31  # reuse
        buf33 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_8, batch_norm_5, out_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf32, primals_41, primals_42, primals_43, primals_44, primals_45, buf33, 65536, grid=grid(65536), stream=stream0)
        del primals_41
        del primals_45
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf35 = buf34; del buf34  # reuse
        buf36 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_9, out_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_3.run(buf35, primals_47, primals_48, primals_49, primals_50, primals_51, buf36, 65536, grid=grid(65536), stream=stream0)
        del primals_47
        del primals_51
        # Topologically Sorted Source Nodes: [context_mask_4], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 1, 16, 16), (256, 256, 16, 1))
        buf40 = reinterpret_tensor(buf37, (4, 1, 256), (256, 256, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [context_mask_6], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_4.run(buf40, primals_53, 4, 256, grid=grid(4), stream=stream0)
        del primals_53
        buf41 = reinterpret_tensor(buf26, (4, 64, 1), (64, 1, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [context_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (4, 64, 256), (16384, 256, 1), 0), reinterpret_tensor(buf40, (4, 256, 1), (256, 1, 1), 0), out=buf41)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(reinterpret_tensor(buf41, (4, 64, 1, 1), (64, 1, 1, 1), 0), primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 128, 1, 1), (128, 1, 1, 1))
        buf43 = buf42; del buf42  # reuse
        buf44 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf45 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf47 = reinterpret_tensor(buf45, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf45  # reuse
        buf48 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7, input_8], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_5.run(buf43, buf47, primals_55, primals_56, primals_57, buf44, buf48, 4, 128, grid=grid(4), stream=stream0)
        del primals_55
        del primals_57
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 64, 1, 1), (64, 1, 1, 1))
        buf50 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, out_8, out_9, input_10], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_6.run(buf36, buf49, primals_59, buf27, buf50, 65536, grid=grid(65536), stream=stream0)
        del primals_59
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf52 = buf51; del buf51  # reuse
        buf53 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_13, batch_norm_7, out_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf52, primals_61, primals_62, primals_63, primals_64, primals_65, buf53, 65536, grid=grid(65536), stream=stream0)
        del primals_61
        del primals_65
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf55 = buf54; del buf54  # reuse
        buf56 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_14, batch_norm_8, out_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf55, primals_67, primals_68, primals_69, primals_70, primals_71, buf56, 65536, grid=grid(65536), stream=stream0)
        del primals_67
        del primals_71
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf58 = buf57; del buf57  # reuse
        buf59 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_15, out_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_3.run(buf58, primals_73, primals_74, primals_75, primals_76, primals_77, buf59, 65536, grid=grid(65536), stream=stream0)
        del primals_73
        del primals_77
        # Topologically Sorted Source Nodes: [context_mask_8], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 1, 16, 16), (256, 256, 16, 1))
        buf63 = reinterpret_tensor(buf60, (4, 1, 256), (256, 256, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [context_mask_10], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_4.run(buf63, primals_79, 4, 256, grid=grid(4), stream=stream0)
        del primals_79
        buf64 = reinterpret_tensor(buf49, (4, 64, 1), (64, 1, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [context_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (4, 64, 256), (16384, 256, 1), 0), reinterpret_tensor(buf63, (4, 256, 1), (256, 1, 1), 0), out=buf64)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(reinterpret_tensor(buf64, (4, 64, 1, 1), (64, 1, 1, 1), 0), primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 128, 1, 1), (128, 1, 1, 1))
        buf66 = buf65; del buf65  # reuse
        buf67 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf68 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf70 = reinterpret_tensor(buf68, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf68  # reuse
        buf71 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_5.run(buf66, buf70, primals_81, primals_82, primals_83, buf67, buf71, 4, 128, grid=grid(4), stream=stream0)
        del primals_81
        del primals_83
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_84, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 64, 1, 1), (64, 1, 1, 1))
        buf73 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, out_13, out_14, input_15], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_6.run(buf59, buf72, primals_85, buf50, buf73, 65536, grid=grid(65536), stream=stream0)
        del buf72
        del primals_85
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf75 = buf74; del buf74  # reuse
        buf76 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_19, batch_norm_10, out_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf75, primals_87, primals_88, primals_89, primals_90, primals_91, buf76, 131072, grid=grid(131072), stream=stream0)
        del primals_87
        del primals_91
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_92, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf78 = buf77; del buf77  # reuse
        buf79 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_20, batch_norm_11, out_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf78, primals_93, primals_94, primals_95, primals_96, primals_97, buf79, 32768, grid=grid(32768), stream=stream0)
        del primals_93
        del primals_97
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf81 = buf80; del buf80  # reuse
        buf82 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_21, out_17], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf81, primals_99, primals_100, primals_101, primals_102, primals_103, buf82, 32768, grid=grid(32768), stream=stream0)
        del primals_103
        del primals_99
        # Topologically Sorted Source Nodes: [context_mask_12], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 1, 8, 8), (64, 64, 8, 1))
        buf86 = reinterpret_tensor(buf83, (4, 1, 64), (64, 64, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [context_mask_14], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_10.run(buf86, primals_105, 4, 64, grid=grid(4), stream=stream0)
        del primals_105
        buf87 = empty_strided_cuda((4, 128, 1), (128, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (4, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf86, (4, 64, 1), (64, 1, 1), 0), out=buf87)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(reinterpret_tensor(buf87, (4, 128, 1, 1), (128, 1, 1, 1), 0), primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 256, 1, 1), (256, 1, 1, 1))
        buf89 = buf88; del buf88  # reuse
        buf90 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf91 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf93 = reinterpret_tensor(buf91, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf91  # reuse
        buf94 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_11.run(buf89, buf93, primals_107, primals_108, primals_109, buf90, buf94, 4, 256, grid=grid(4), stream=stream0)
        del primals_107
        del primals_109
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 128, 1, 1), (128, 1, 1, 1))
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf73, primals_112, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf97 = buf96; del buf96  # reuse
        buf98 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, out_18, input_20, input_21, out_19, input_22], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12.run(buf97, primals_113, buf82, buf95, primals_111, primals_114, primals_115, primals_116, primals_117, buf98, 32768, grid=grid(32768), stream=stream0)
        del primals_111
        del primals_113
        del primals_117
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf100 = buf99; del buf99  # reuse
        buf101 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_26, batch_norm_14, out_20], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf100, primals_119, primals_120, primals_121, primals_122, primals_123, buf101, 32768, grid=grid(32768), stream=stream0)
        del primals_119
        del primals_123
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf103 = buf102; del buf102  # reuse
        buf104 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_27, batch_norm_15, out_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf103, primals_125, primals_126, primals_127, primals_128, primals_129, buf104, 32768, grid=grid(32768), stream=stream0)
        del primals_125
        del primals_129
        # Topologically Sorted Source Nodes: [conv2d_28], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf106 = buf105; del buf105  # reuse
        buf107 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_28, out_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf106, primals_131, primals_132, primals_133, primals_134, primals_135, buf107, 32768, grid=grid(32768), stream=stream0)
        del primals_131
        del primals_135
        # Topologically Sorted Source Nodes: [context_mask_16], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 1, 8, 8), (64, 64, 8, 1))
        buf111 = reinterpret_tensor(buf108, (4, 1, 64), (64, 64, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [context_mask_18], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_10.run(buf111, primals_137, 4, 64, grid=grid(4), stream=stream0)
        del primals_137
        buf112 = reinterpret_tensor(buf95, (4, 128, 1), (128, 1, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [context_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (4, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf111, (4, 64, 1), (64, 1, 1), 0), out=buf112)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(reinterpret_tensor(buf112, (4, 128, 1, 1), (128, 1, 1, 1), 0), primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 256, 1, 1), (256, 1, 1, 1))
        buf114 = buf113; del buf113  # reuse
        buf115 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf116 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf118 = reinterpret_tensor(buf116, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf116  # reuse
        buf119 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23, input_24, input_25], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_11.run(buf114, buf118, primals_139, primals_140, primals_141, buf115, buf119, 4, 256, grid=grid(4), stream=stream0)
        del primals_139
        del primals_141
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 128, 1, 1), (128, 1, 1, 1))
        buf121 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, out_23, out_24, input_27], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_13.run(buf107, buf120, primals_143, buf98, buf121, 32768, grid=grid(32768), stream=stream0)
        del primals_143
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf123 = buf122; del buf122  # reuse
        buf124 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_32, batch_norm_17, out_25], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf123, primals_145, primals_146, primals_147, primals_148, primals_149, buf124, 32768, grid=grid(32768), stream=stream0)
        del primals_145
        del primals_149
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf126 = buf125; del buf125  # reuse
        buf127 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_33, batch_norm_18, out_26], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf126, primals_151, primals_152, primals_153, primals_154, primals_155, buf127, 32768, grid=grid(32768), stream=stream0)
        del primals_151
        del primals_155
        # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf129 = buf128; del buf128  # reuse
        buf130 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_34, out_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf129, primals_157, primals_158, primals_159, primals_160, primals_161, buf130, 32768, grid=grid(32768), stream=stream0)
        del primals_157
        del primals_161
        # Topologically Sorted Source Nodes: [context_mask_20], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 1, 8, 8), (64, 64, 8, 1))
        buf134 = reinterpret_tensor(buf131, (4, 1, 64), (64, 64, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [context_mask_22], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_10.run(buf134, primals_163, 4, 64, grid=grid(4), stream=stream0)
        del primals_163
        buf135 = reinterpret_tensor(buf120, (4, 128, 1), (128, 1, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [context_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf130, (4, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf134, (4, 64, 1), (64, 1, 1), 0), out=buf135)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(reinterpret_tensor(buf135, (4, 128, 1, 1), (128, 1, 1, 1), 0), primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 256, 1, 1), (256, 1, 1, 1))
        buf137 = buf136; del buf136  # reuse
        buf138 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf139 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf141 = reinterpret_tensor(buf139, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf139  # reuse
        buf142 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, input_29, input_30], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_11.run(buf137, buf141, primals_165, primals_166, primals_167, buf138, buf142, 4, 256, grid=grid(4), stream=stream0)
        del primals_165
        del primals_167
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 128, 1, 1), (128, 1, 1, 1))
        buf144 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31, out_28, out_29, input_32], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_13.run(buf130, buf143, primals_169, buf121, buf144, 32768, grid=grid(32768), stream=stream0)
        del primals_169
        # Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf146 = buf145; del buf145  # reuse
        buf147 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_38, batch_norm_20, out_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf146, primals_171, primals_172, primals_173, primals_174, primals_175, buf147, 32768, grid=grid(32768), stream=stream0)
        del primals_171
        del primals_175
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf149 = buf148; del buf148  # reuse
        buf150 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_39, batch_norm_21, out_31], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf149, primals_177, primals_178, primals_179, primals_180, primals_181, buf150, 32768, grid=grid(32768), stream=stream0)
        del primals_177
        del primals_181
        # Topologically Sorted Source Nodes: [conv2d_40], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf152 = buf151; del buf151  # reuse
        buf153 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_40, out_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf152, primals_183, primals_184, primals_185, primals_186, primals_187, buf153, 32768, grid=grid(32768), stream=stream0)
        del primals_183
        del primals_187
        # Topologically Sorted Source Nodes: [context_mask_24], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 1, 8, 8), (64, 64, 8, 1))
        buf157 = reinterpret_tensor(buf154, (4, 1, 64), (64, 64, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [context_mask_26], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_10.run(buf157, primals_189, 4, 64, grid=grid(4), stream=stream0)
        del primals_189
        buf158 = reinterpret_tensor(buf143, (4, 128, 1), (128, 1, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [context_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (4, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf157, (4, 64, 1), (64, 1, 1), 0), out=buf158)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(reinterpret_tensor(buf158, (4, 128, 1, 1), (128, 1, 1, 1), 0), primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 256, 1, 1), (256, 1, 1, 1))
        buf160 = buf159; del buf159  # reuse
        buf161 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf162 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf164 = reinterpret_tensor(buf162, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf162  # reuse
        buf165 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_33, input_34, input_35], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_11.run(buf160, buf164, primals_191, primals_192, primals_193, buf161, buf165, 4, 256, grid=grid(4), stream=stream0)
        del primals_191
        del primals_193
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 128, 1, 1), (128, 1, 1, 1))
        buf167 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_36, out_33, out_34, input_37], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_13.run(buf153, buf166, primals_195, buf144, buf167, 32768, grid=grid(32768), stream=stream0)
        del buf166
        del primals_195
        # Topologically Sorted Source Nodes: [conv2d_44], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf169 = buf168; del buf168  # reuse
        buf170 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_44, batch_norm_23, out_35], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf169, primals_197, primals_198, primals_199, primals_200, primals_201, buf170, 65536, grid=grid(65536), stream=stream0)
        del primals_197
        del primals_201
        # Topologically Sorted Source Nodes: [conv2d_45], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_202, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf172 = buf171; del buf171  # reuse
        buf173 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_45, batch_norm_24, out_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf172, primals_203, primals_204, primals_205, primals_206, primals_207, buf173, 16384, grid=grid(16384), stream=stream0)
        del primals_203
        del primals_207
        # Topologically Sorted Source Nodes: [conv2d_46], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf175 = buf174; del buf174  # reuse
        buf176 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_46, out_37], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_16.run(buf175, primals_209, primals_210, primals_211, primals_212, primals_213, buf176, 16384, grid=grid(16384), stream=stream0)
        del primals_209
        del primals_213
        # Topologically Sorted Source Nodes: [context_mask_28], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 1, 4, 4), (16, 16, 4, 1))
        buf180 = reinterpret_tensor(buf177, (4, 1, 16), (16, 16, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [context_mask_30], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_17.run(buf180, primals_215, 4, 16, grid=grid(4), stream=stream0)
        del primals_215
        buf181 = empty_strided_cuda((4, 256, 1), (256, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (4, 256, 16), (4096, 16, 1), 0), reinterpret_tensor(buf180, (4, 16, 1), (16, 1, 1), 0), out=buf181)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(reinterpret_tensor(buf181, (4, 256, 1, 1), (256, 1, 1, 1), 0), primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 512, 1, 1), (512, 1, 1, 1))
        buf183 = buf182; del buf182  # reuse
        buf184 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf185 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf187 = reinterpret_tensor(buf185, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf185  # reuse
        buf188 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_18.run(buf183, buf187, primals_217, primals_218, primals_219, buf184, buf188, 4, 512, grid=grid(4), stream=stream0)
        del primals_217
        del primals_219
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 256, 1, 1), (256, 1, 1, 1))
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf167, primals_222, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf191 = buf190; del buf190  # reuse
        buf192 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_41, out_38, input_42, input_43, out_39, input_44], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19.run(buf191, primals_223, buf176, buf189, primals_221, primals_224, primals_225, primals_226, primals_227, buf192, 16384, grid=grid(16384), stream=stream0)
        del primals_221
        del primals_223
        del primals_227
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf194 = buf193; del buf193  # reuse
        buf195 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_51, batch_norm_27, out_40], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf194, primals_229, primals_230, primals_231, primals_232, primals_233, buf195, 16384, grid=grid(16384), stream=stream0)
        del primals_229
        del primals_233
        # Topologically Sorted Source Nodes: [conv2d_52], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_234, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf197 = buf196; del buf196  # reuse
        buf198 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_52, batch_norm_28, out_41], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf197, primals_235, primals_236, primals_237, primals_238, primals_239, buf198, 16384, grid=grid(16384), stream=stream0)
        del primals_235
        del primals_239
        # Topologically Sorted Source Nodes: [conv2d_53], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf200 = buf199; del buf199  # reuse
        buf201 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_53, out_42], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_16.run(buf200, primals_241, primals_242, primals_243, primals_244, primals_245, buf201, 16384, grid=grid(16384), stream=stream0)
        del primals_241
        del primals_245
        # Topologically Sorted Source Nodes: [context_mask_32], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 1, 4, 4), (16, 16, 4, 1))
        buf205 = reinterpret_tensor(buf202, (4, 1, 16), (16, 16, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [context_mask_34], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_17.run(buf205, primals_247, 4, 16, grid=grid(4), stream=stream0)
        del primals_247
        buf206 = reinterpret_tensor(buf189, (4, 256, 1), (256, 1, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [context_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (4, 256, 16), (4096, 16, 1), 0), reinterpret_tensor(buf205, (4, 16, 1), (16, 1, 1), 0), out=buf206)
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(reinterpret_tensor(buf206, (4, 256, 1, 1), (256, 1, 1, 1), 0), primals_248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 512, 1, 1), (512, 1, 1, 1))
        buf208 = buf207; del buf207  # reuse
        buf209 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf210 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf212 = reinterpret_tensor(buf210, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf210  # reuse
        buf213 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_45, input_46, input_47], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_18.run(buf208, buf212, primals_249, primals_250, primals_251, buf209, buf213, 4, 512, grid=grid(4), stream=stream0)
        del primals_249
        del primals_251
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 256, 1, 1), (256, 1, 1, 1))
        buf215 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_48, out_43, out_44, input_49], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_20.run(buf201, buf214, primals_253, buf192, buf215, 16384, grid=grid(16384), stream=stream0)
        del primals_253
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf217 = buf216; del buf216  # reuse
        buf218 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_57, batch_norm_30, out_45], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf217, primals_255, primals_256, primals_257, primals_258, primals_259, buf218, 16384, grid=grid(16384), stream=stream0)
        del primals_255
        del primals_259
        # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_260, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf220 = buf219; del buf219  # reuse
        buf221 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_58, batch_norm_31, out_46], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf220, primals_261, primals_262, primals_263, primals_264, primals_265, buf221, 16384, grid=grid(16384), stream=stream0)
        del primals_261
        del primals_265
        # Topologically Sorted Source Nodes: [conv2d_59], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_266, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf223 = buf222; del buf222  # reuse
        buf224 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_59, out_47], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_16.run(buf223, primals_267, primals_268, primals_269, primals_270, primals_271, buf224, 16384, grid=grid(16384), stream=stream0)
        del primals_267
        del primals_271
        # Topologically Sorted Source Nodes: [context_mask_36], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 1, 4, 4), (16, 16, 4, 1))
        buf228 = reinterpret_tensor(buf225, (4, 1, 16), (16, 16, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [context_mask_38], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_17.run(buf228, primals_273, 4, 16, grid=grid(4), stream=stream0)
        del primals_273
        buf229 = reinterpret_tensor(buf214, (4, 256, 1), (256, 1, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [context_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (4, 256, 16), (4096, 16, 1), 0), reinterpret_tensor(buf228, (4, 16, 1), (16, 1, 1), 0), out=buf229)
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(reinterpret_tensor(buf229, (4, 256, 1, 1), (256, 1, 1, 1), 0), primals_274, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 512, 1, 1), (512, 1, 1, 1))
        buf231 = buf230; del buf230  # reuse
        buf232 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf233 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf235 = reinterpret_tensor(buf233, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf233  # reuse
        buf236 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_50, input_51, input_52], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_18.run(buf231, buf235, primals_275, primals_276, primals_277, buf232, buf236, 4, 512, grid=grid(4), stream=stream0)
        del primals_275
        del primals_277
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_278, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 256, 1, 1), (256, 1, 1, 1))
        buf238 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_53, out_48, out_49, input_54], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_20.run(buf224, buf237, primals_279, buf215, buf238, 16384, grid=grid(16384), stream=stream0)
        del primals_279
        # Topologically Sorted Source Nodes: [conv2d_63], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf240 = buf239; del buf239  # reuse
        buf241 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_63, batch_norm_33, out_50], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf240, primals_281, primals_282, primals_283, primals_284, primals_285, buf241, 16384, grid=grid(16384), stream=stream0)
        del primals_281
        del primals_285
        # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_286, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf243 = buf242; del buf242  # reuse
        buf244 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_64, batch_norm_34, out_51], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf243, primals_287, primals_288, primals_289, primals_290, primals_291, buf244, 16384, grid=grid(16384), stream=stream0)
        del primals_287
        del primals_291
        # Topologically Sorted Source Nodes: [conv2d_65], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf246 = buf245; del buf245  # reuse
        buf247 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_65, out_52], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_16.run(buf246, primals_293, primals_294, primals_295, primals_296, primals_297, buf247, 16384, grid=grid(16384), stream=stream0)
        del primals_293
        del primals_297
        # Topologically Sorted Source Nodes: [context_mask_40], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_298, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 1, 4, 4), (16, 16, 4, 1))
        buf251 = reinterpret_tensor(buf248, (4, 1, 16), (16, 16, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [context_mask_42], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_17.run(buf251, primals_299, 4, 16, grid=grid(4), stream=stream0)
        del primals_299
        buf252 = reinterpret_tensor(buf237, (4, 256, 1), (256, 1, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [context_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (4, 256, 16), (4096, 16, 1), 0), reinterpret_tensor(buf251, (4, 16, 1), (16, 1, 1), 0), out=buf252)
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(reinterpret_tensor(buf252, (4, 256, 1, 1), (256, 1, 1, 1), 0), primals_300, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 512, 1, 1), (512, 1, 1, 1))
        buf254 = buf253; del buf253  # reuse
        buf255 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf256 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf258 = reinterpret_tensor(buf256, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf256  # reuse
        buf259 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_55, input_56, input_57], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_18.run(buf254, buf258, primals_301, primals_302, primals_303, buf255, buf259, 4, 512, grid=grid(4), stream=stream0)
        del primals_301
        del primals_303
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 256, 1, 1), (256, 1, 1, 1))
        buf261 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_58, out_53, out_54, input_59], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_20.run(buf247, buf260, primals_305, buf238, buf261, 16384, grid=grid(16384), stream=stream0)
        del primals_305
        # Topologically Sorted Source Nodes: [conv2d_69], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_306, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf263 = buf262; del buf262  # reuse
        buf264 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_69, batch_norm_36, out_55], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf263, primals_307, primals_308, primals_309, primals_310, primals_311, buf264, 16384, grid=grid(16384), stream=stream0)
        del primals_307
        del primals_311
        # Topologically Sorted Source Nodes: [conv2d_70], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf266 = buf265; del buf265  # reuse
        buf267 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_70, batch_norm_37, out_56], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf266, primals_313, primals_314, primals_315, primals_316, primals_317, buf267, 16384, grid=grid(16384), stream=stream0)
        del primals_313
        del primals_317
        # Topologically Sorted Source Nodes: [conv2d_71], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_318, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf269 = buf268; del buf268  # reuse
        buf270 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_71, out_57], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_16.run(buf269, primals_319, primals_320, primals_321, primals_322, primals_323, buf270, 16384, grid=grid(16384), stream=stream0)
        del primals_319
        del primals_323
        # Topologically Sorted Source Nodes: [context_mask_44], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_324, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 1, 4, 4), (16, 16, 4, 1))
        buf274 = reinterpret_tensor(buf271, (4, 1, 16), (16, 16, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [context_mask_46], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_17.run(buf274, primals_325, 4, 16, grid=grid(4), stream=stream0)
        del primals_325
        buf275 = reinterpret_tensor(buf260, (4, 256, 1), (256, 1, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [context_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (4, 256, 16), (4096, 16, 1), 0), reinterpret_tensor(buf274, (4, 16, 1), (16, 1, 1), 0), out=buf275)
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(reinterpret_tensor(buf275, (4, 256, 1, 1), (256, 1, 1, 1), 0), primals_326, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 512, 1, 1), (512, 1, 1, 1))
        buf277 = buf276; del buf276  # reuse
        buf278 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf279 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf281 = reinterpret_tensor(buf279, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf279  # reuse
        buf282 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_60, input_61, input_62], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_18.run(buf277, buf281, primals_327, primals_328, primals_329, buf278, buf282, 4, 512, grid=grid(4), stream=stream0)
        del primals_327
        del primals_329
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_330, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 256, 1, 1), (256, 1, 1, 1))
        buf284 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_63, out_58, out_59, input_64], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_20.run(buf270, buf283, primals_331, buf261, buf284, 16384, grid=grid(16384), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [conv2d_75], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf286 = buf285; del buf285  # reuse
        buf287 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_75, batch_norm_39, out_60], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf286, primals_333, primals_334, primals_335, primals_336, primals_337, buf287, 16384, grid=grid(16384), stream=stream0)
        del primals_333
        del primals_337
        # Topologically Sorted Source Nodes: [conv2d_76], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_338, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf289 = buf288; del buf288  # reuse
        buf290 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_76, batch_norm_40, out_61], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf289, primals_339, primals_340, primals_341, primals_342, primals_343, buf290, 16384, grid=grid(16384), stream=stream0)
        del primals_339
        del primals_343
        # Topologically Sorted Source Nodes: [conv2d_77], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_344, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf292 = buf291; del buf291  # reuse
        buf293 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_77, out_62], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_16.run(buf292, primals_345, primals_346, primals_347, primals_348, primals_349, buf293, 16384, grid=grid(16384), stream=stream0)
        del primals_345
        del primals_349
        # Topologically Sorted Source Nodes: [context_mask_48], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_350, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 1, 4, 4), (16, 16, 4, 1))
        buf297 = reinterpret_tensor(buf294, (4, 1, 16), (16, 16, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [context_mask_50], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_17.run(buf297, primals_351, 4, 16, grid=grid(4), stream=stream0)
        del primals_351
        buf298 = reinterpret_tensor(buf283, (4, 256, 1), (256, 1, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [context_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf293, (4, 256, 16), (4096, 16, 1), 0), reinterpret_tensor(buf297, (4, 16, 1), (16, 1, 1), 0), out=buf298)
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(reinterpret_tensor(buf298, (4, 256, 1, 1), (256, 1, 1, 1), 0), primals_352, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 512, 1, 1), (512, 1, 1, 1))
        buf300 = buf299; del buf299  # reuse
        buf301 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf302 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf304 = reinterpret_tensor(buf302, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf302  # reuse
        buf305 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_65, input_66, input_67], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_18.run(buf300, buf304, primals_353, primals_354, primals_355, buf301, buf305, 4, 512, grid=grid(4), stream=stream0)
        del primals_353
        del primals_355
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_356, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 256, 1, 1), (256, 1, 1, 1))
        buf307 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_68, out_63, out_64, input_69], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_20.run(buf293, buf306, primals_357, buf284, buf307, 16384, grid=grid(16384), stream=stream0)
        del buf306
        del primals_357
        # Topologically Sorted Source Nodes: [conv2d_81], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_358, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf309 = buf308; del buf308  # reuse
        buf310 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_81, batch_norm_42, out_65], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf309, primals_359, primals_360, primals_361, primals_362, primals_363, buf310, 32768, grid=grid(32768), stream=stream0)
        del primals_359
        del primals_363
        # Topologically Sorted Source Nodes: [conv2d_82], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, primals_364, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf312 = buf311; del buf311  # reuse
        buf313 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_82, batch_norm_43, out_66], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf312, primals_365, primals_366, primals_367, primals_368, primals_369, buf313, 32768, grid=grid(32768), stream=stream0)
        del primals_365
        del primals_369
        # Topologically Sorted Source Nodes: [conv2d_83], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, primals_370, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf315 = buf314; del buf314  # reuse
        buf316 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_83, out_67], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_22.run(buf315, primals_371, primals_372, primals_373, primals_374, primals_375, buf316, 32768, grid=grid(32768), stream=stream0)
        del primals_371
        del primals_375
        # Topologically Sorted Source Nodes: [context_mask_52], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_376, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 1, 4, 4), (16, 16, 4, 1))
        buf320 = reinterpret_tensor(buf317, (4, 1, 16), (16, 16, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [context_mask_54], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_17.run(buf320, primals_377, 4, 16, grid=grid(4), stream=stream0)
        del primals_377
        buf321 = empty_strided_cuda((4, 512, 1), (512, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf316, (4, 512, 16), (8192, 16, 1), 0), reinterpret_tensor(buf320, (4, 16, 1), (16, 1, 1), 0), out=buf321)
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(reinterpret_tensor(buf321, (4, 512, 1, 1), (512, 1, 1, 1), 0), primals_378, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 1024, 1, 1), (1024, 1, 1, 1))
        buf323 = buf322; del buf322  # reuse
        buf324 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf325 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf327 = reinterpret_tensor(buf325, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf325  # reuse
        buf328 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_70, input_71, input_72], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_23.run(buf323, buf327, primals_379, primals_380, primals_381, buf324, buf328, 4, 1024, grid=grid(4), stream=stream0)
        del primals_379
        del primals_381
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_382, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (4, 512, 1, 1), (512, 1, 1, 1))
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf307, primals_384, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf331 = buf330; del buf330  # reuse
        buf332 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_73, out_68, input_74, input_75, out_69, input_76], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(buf331, primals_385, buf316, buf329, primals_383, primals_386, primals_387, primals_388, primals_389, buf332, 32768, grid=grid(32768), stream=stream0)
        del primals_383
        del primals_385
        del primals_389
        # Topologically Sorted Source Nodes: [conv2d_88], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_390, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf334 = buf333; del buf333  # reuse
        buf335 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_88, batch_norm_46, out_70], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf334, primals_391, primals_392, primals_393, primals_394, primals_395, buf335, 32768, grid=grid(32768), stream=stream0)
        del primals_391
        del primals_395
        # Topologically Sorted Source Nodes: [conv2d_89], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_396, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf337 = buf336; del buf336  # reuse
        buf338 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_89, batch_norm_47, out_71], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf337, primals_397, primals_398, primals_399, primals_400, primals_401, buf338, 32768, grid=grid(32768), stream=stream0)
        del primals_397
        del primals_401
        # Topologically Sorted Source Nodes: [conv2d_90], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf340 = buf339; del buf339  # reuse
        buf341 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_90, out_72], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_22.run(buf340, primals_403, primals_404, primals_405, primals_406, primals_407, buf341, 32768, grid=grid(32768), stream=stream0)
        del primals_403
        del primals_407
        # Topologically Sorted Source Nodes: [context_mask_56], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_408, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 1, 4, 4), (16, 16, 4, 1))
        buf345 = reinterpret_tensor(buf342, (4, 1, 16), (16, 16, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [context_mask_58], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_17.run(buf345, primals_409, 4, 16, grid=grid(4), stream=stream0)
        del primals_409
        buf346 = reinterpret_tensor(buf329, (4, 512, 1), (512, 1, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [context_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf341, (4, 512, 16), (8192, 16, 1), 0), reinterpret_tensor(buf345, (4, 16, 1), (16, 1, 1), 0), out=buf346)
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(reinterpret_tensor(buf346, (4, 512, 1, 1), (512, 1, 1, 1), 0), primals_410, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (4, 1024, 1, 1), (1024, 1, 1, 1))
        buf348 = buf347; del buf347  # reuse
        buf349 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf350 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf352 = reinterpret_tensor(buf350, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf350  # reuse
        buf353 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_77, input_78, input_79], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_23.run(buf348, buf352, primals_411, primals_412, primals_413, buf349, buf353, 4, 1024, grid=grid(4), stream=stream0)
        del primals_411
        del primals_413
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_414, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 512, 1, 1), (512, 1, 1, 1))
        buf355 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80, out_73, out_74, input_81], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_25.run(buf341, buf354, primals_415, buf332, buf355, 32768, grid=grid(32768), stream=stream0)
        del primals_415
        # Topologically Sorted Source Nodes: [conv2d_94], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_416, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf357 = buf356; del buf356  # reuse
        buf358 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_94, batch_norm_49, out_75], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf357, primals_417, primals_418, primals_419, primals_420, primals_421, buf358, 32768, grid=grid(32768), stream=stream0)
        del primals_417
        del primals_421
        # Topologically Sorted Source Nodes: [conv2d_95], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf360 = buf359; del buf359  # reuse
        buf361 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_95, batch_norm_50, out_76], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf360, primals_423, primals_424, primals_425, primals_426, primals_427, buf361, 32768, grid=grid(32768), stream=stream0)
        del primals_423
        del primals_427
        # Topologically Sorted Source Nodes: [conv2d_96], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_428, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf363 = buf362; del buf362  # reuse
        buf364 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_96, out_77], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_22.run(buf363, primals_429, primals_430, primals_431, primals_432, primals_433, buf364, 32768, grid=grid(32768), stream=stream0)
        del primals_429
        del primals_433
        # Topologically Sorted Source Nodes: [context_mask_60], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_434, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (4, 1, 4, 4), (16, 16, 4, 1))
        buf368 = reinterpret_tensor(buf365, (4, 1, 16), (16, 16, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [context_mask_62], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_17.run(buf368, primals_435, 4, 16, grid=grid(4), stream=stream0)
        del primals_435
        buf369 = reinterpret_tensor(buf354, (4, 512, 1), (512, 1, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [context_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf364, (4, 512, 16), (8192, 16, 1), 0), reinterpret_tensor(buf368, (4, 16, 1), (16, 1, 1), 0), out=buf369)
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(reinterpret_tensor(buf369, (4, 512, 1, 1), (512, 1, 1, 1), 0), primals_436, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (4, 1024, 1, 1), (1024, 1, 1, 1))
        buf371 = buf370; del buf370  # reuse
        buf372 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf373 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf375 = reinterpret_tensor(buf373, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf373  # reuse
        buf376 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_82, input_83, input_84], Original ATen: [aten.convolution, aten.native_layer_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_layer_norm_relu_23.run(buf371, buf375, primals_437, primals_438, primals_439, buf372, buf376, 4, 1024, grid=grid(4), stream=stream0)
        del primals_437
        del primals_439
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_440, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 512, 1, 1), (512, 1, 1, 1))
        buf378 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_85, out_78, out_79, input_86], Original ATen: [aten.convolution, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_25.run(buf364, buf377, primals_441, buf355, buf378, 32768, grid=grid(32768), stream=stream0)
        del buf377
        del primals_441
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_442, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf379, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf380 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_26.run(buf380, primals_443, 65536, grid=grid(65536), stream=stream0)
        del primals_443
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, primals_444, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf382 = buf381; del buf381  # reuse
        buf383 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_88, input_89, input_90], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf382, primals_445, primals_446, primals_447, primals_448, primals_449, buf383, 32768, grid=grid(32768), stream=stream0)
        del primals_445
        del primals_449
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf384 = extern_kernels.convolution(buf383, primals_450, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf384, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf385 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_27.run(buf385, primals_451, 131072, grid=grid(131072), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [input_92], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, primals_452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf387 = buf386; del buf386  # reuse
        buf388 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_92, input_93, input_94], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf387, primals_453, primals_454, primals_455, primals_456, primals_457, buf388, 65536, grid=grid(65536), stream=stream0)
        del primals_453
        del primals_457
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_458, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf390 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf390, primals_459, 4096, grid=grid(4096), stream=stream0)
        del primals_459
    return (buf390, primals_1, primals_3, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_37, primals_38, primals_40, primals_42, primals_43, primals_44, primals_46, primals_48, primals_49, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_63, primals_64, primals_66, primals_68, primals_69, primals_70, primals_72, primals_74, primals_75, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_98, primals_100, primals_101, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_115, primals_116, primals_118, primals_120, primals_121, primals_122, primals_124, primals_126, primals_127, primals_128, primals_130, primals_132, primals_133, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_147, primals_148, primals_150, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_173, primals_174, primals_176, primals_178, primals_179, primals_180, primals_182, primals_184, primals_185, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_199, primals_200, primals_202, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, primals_212, primals_214, primals_216, primals_218, primals_220, primals_222, primals_224, primals_225, primals_226, primals_228, primals_230, primals_231, primals_232, primals_234, primals_236, primals_237, primals_238, primals_240, primals_242, primals_243, primals_244, primals_246, primals_248, primals_250, primals_252, primals_254, primals_256, primals_257, primals_258, primals_260, primals_262, primals_263, primals_264, primals_266, primals_268, primals_269, primals_270, primals_272, primals_274, primals_276, primals_278, primals_280, primals_282, primals_283, primals_284, primals_286, primals_288, primals_289, primals_290, primals_292, primals_294, primals_295, primals_296, primals_298, primals_300, primals_302, primals_304, primals_306, primals_308, primals_309, primals_310, primals_312, primals_314, primals_315, primals_316, primals_318, primals_320, primals_321, primals_322, primals_324, primals_326, primals_328, primals_330, primals_332, primals_334, primals_335, primals_336, primals_338, primals_340, primals_341, primals_342, primals_344, primals_346, primals_347, primals_348, primals_350, primals_352, primals_354, primals_356, primals_358, primals_360, primals_361, primals_362, primals_364, primals_366, primals_367, primals_368, primals_370, primals_372, primals_373, primals_374, primals_376, primals_378, primals_380, primals_382, primals_384, primals_386, primals_387, primals_388, primals_390, primals_392, primals_393, primals_394, primals_396, primals_398, primals_399, primals_400, primals_402, primals_404, primals_405, primals_406, primals_408, primals_410, primals_412, primals_414, primals_416, primals_418, primals_419, primals_420, primals_422, primals_424, primals_425, primals_426, primals_428, primals_430, primals_431, primals_432, primals_434, primals_436, primals_438, primals_440, primals_442, primals_444, primals_446, primals_447, primals_448, primals_450, primals_452, primals_454, primals_455, primals_456, primals_458, buf1, buf2, buf3, buf4, buf6, buf7, buf9, buf10, buf12, buf13, buf17, reinterpret_tensor(buf18, (4, 64, 1, 1), (64, 1, 1, 1), 0), buf20, buf21, buf24, buf25, buf27, buf29, buf30, buf32, buf33, buf35, buf36, buf40, reinterpret_tensor(buf41, (4, 64, 1, 1), (64, 1, 1, 1), 0), buf43, buf44, buf47, buf48, buf50, buf52, buf53, buf55, buf56, buf58, buf59, buf63, reinterpret_tensor(buf64, (4, 64, 1, 1), (64, 1, 1, 1), 0), buf66, buf67, buf70, buf71, buf73, buf75, buf76, buf78, buf79, buf81, buf82, buf86, reinterpret_tensor(buf87, (4, 128, 1, 1), (128, 1, 1, 1), 0), buf89, buf90, buf93, buf94, buf97, buf98, buf100, buf101, buf103, buf104, buf106, buf107, buf111, reinterpret_tensor(buf112, (4, 128, 1, 1), (128, 1, 1, 1), 0), buf114, buf115, buf118, buf119, buf121, buf123, buf124, buf126, buf127, buf129, buf130, buf134, reinterpret_tensor(buf135, (4, 128, 1, 1), (128, 1, 1, 1), 0), buf137, buf138, buf141, buf142, buf144, buf146, buf147, buf149, buf150, buf152, buf153, buf157, reinterpret_tensor(buf158, (4, 128, 1, 1), (128, 1, 1, 1), 0), buf160, buf161, buf164, buf165, buf167, buf169, buf170, buf172, buf173, buf175, buf176, buf180, reinterpret_tensor(buf181, (4, 256, 1, 1), (256, 1, 1, 1), 0), buf183, buf184, buf187, buf188, buf191, buf192, buf194, buf195, buf197, buf198, buf200, buf201, buf205, reinterpret_tensor(buf206, (4, 256, 1, 1), (256, 1, 1, 1), 0), buf208, buf209, buf212, buf213, buf215, buf217, buf218, buf220, buf221, buf223, buf224, buf228, reinterpret_tensor(buf229, (4, 256, 1, 1), (256, 1, 1, 1), 0), buf231, buf232, buf235, buf236, buf238, buf240, buf241, buf243, buf244, buf246, buf247, buf251, reinterpret_tensor(buf252, (4, 256, 1, 1), (256, 1, 1, 1), 0), buf254, buf255, buf258, buf259, buf261, buf263, buf264, buf266, buf267, buf269, buf270, buf274, reinterpret_tensor(buf275, (4, 256, 1, 1), (256, 1, 1, 1), 0), buf277, buf278, buf281, buf282, buf284, buf286, buf287, buf289, buf290, buf292, buf293, buf297, reinterpret_tensor(buf298, (4, 256, 1, 1), (256, 1, 1, 1), 0), buf300, buf301, buf304, buf305, buf307, buf309, buf310, buf312, buf313, buf315, buf316, buf320, reinterpret_tensor(buf321, (4, 512, 1, 1), (512, 1, 1, 1), 0), buf323, buf324, buf327, buf328, buf331, buf332, buf334, buf335, buf337, buf338, buf340, buf341, buf345, reinterpret_tensor(buf346, (4, 512, 1, 1), (512, 1, 1, 1), 0), buf348, buf349, buf352, buf353, buf355, buf357, buf358, buf360, buf361, buf363, buf364, buf368, reinterpret_tensor(buf369, (4, 512, 1, 1), (512, 1, 1, 1), 0), buf371, buf372, buf375, buf376, buf378, buf380, buf382, buf383, buf385, buf387, buf388, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((512, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((128, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
