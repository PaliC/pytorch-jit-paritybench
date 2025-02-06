# AOT ID: ['57_forward']
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


# kernel path: inductor_cache/tr/ctrmviqjpqxwffazopoe4qf4og4tvjr2otyin5oktvjyx4rkm66v.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
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
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
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


# kernel path: inductor_cache/fb/cfbrdquw7vtwtfmjklut2fk347kpenecgihcdwp7cifowhsxw7ya.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x => getitem, getitem_1
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
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
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


# kernel path: inductor_cache/se/cseonrqnu5vqexntu435b7wl2ha4ece45sftpmkzssrq6ubhcbxf.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_5 => add_3, mul_4, mul_5, sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ka/cka7stggd45kodklqcmolvcr2ujtgf4mudptmazxhy4dcpqcljxy.py
# Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_7, mul_10, mul_11, sub_3
#   input_11 => relu_2
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gd/cgdew6hueckkhpj23fqlxvxlqtgd3vw5ikeiwxglsgdpfh7g2lqd.py
# Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_13 => add_9, mul_13, mul_14, sub_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m7/cm7qbpfyihj4evambmlrymftgto6yfuuxugfekj24r6nnbqx4l6i.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_1, %relu_3], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 4)
    x0 = (xindex % 64)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 128*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 4, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 64*((-2) + x1) + 128*x2), tmp25 & xmask, other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yg/cygqzn26tlm47qdbunjzv5vnkhpxweegwx5oxoz4qwrv5wecafbe.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_2 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 2)
    x2 = ((xindex // 128) % 2)
    x3 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 128*x1 + 256*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pd/cpdrui3fsci4xon4rjx3tvhvxz7sslvoq3frxdvm4siagbjloh7h.py
# Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_18 => add_13, mul_19, mul_20, sub_6
#   input_19 => relu_4
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5e/c5etsppfzdn72wgqkii5ewt6szf42j7o7rk56362jtj6jgfl6yfw.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_5 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 2)
    x2 = ((xindex // 128) % 2)
    x0 = (xindex % 64)
    x3 = xindex // 256
    x4 = xindex
    tmp0 = x2 + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x2 + 2*x1) + 256*x3), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 4, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-2) + x2 + 2*x1) + 128*x3), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x4), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bk/cbki4tolopxdbzn6gfcgkioph6pvd6vnqm3awmvlwpkle2fc32vc.py
# Topologically Sorted Source Nodes: [input_42], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_42 => add_31, mul_46, mul_47, sub_15
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ub/cubun36ryyqjoen2v33efd3ps76k3h2dbhhpbecrvdbahie5ql2s.py
# Topologically Sorted Source Nodes: [input_50], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_50 => add_37, mul_55, mul_56, sub_18
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_37 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b4/cb4b7efohrxdin5hir5vjyvc5bjxk5mwqzeqxiof7r37x44nphcj.py
# Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_4 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_10, %relu_12], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 4)
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 32*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 4, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 16*((-2) + x1) + 32*x2), tmp25 & xmask, other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x5/cx54atyvxkhni5xgo72decgzl2sdhuflolals3nlydu25cizvry7.py
# Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_14 => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_12 = async_compile.triton('triton_poi_fused_clone_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 2)
    x2 = ((xindex // 32) % 2)
    x3 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 32*x1 + 64*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7g/c7ggtepyl3j4kezthsocgfhjfacgx3wxnzvzuvzj4ezyonzcksqt.py
# Topologically Sorted Source Nodes: [input_55, input_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_55 => add_41, mul_61, mul_62, sub_20
#   input_56 => relu_13
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/43/c43zfjzlq3jlmrvouzraxbdxr4htupt6lcfiuikiq2bodumgnjwb.py
# Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_17 => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_14 = async_compile.triton('triton_poi_fused_clone_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 2)
    x2 = ((xindex // 32) % 2)
    x0 = (xindex % 16)
    x3 = xindex // 64
    x4 = xindex
    tmp0 = x2 + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x2 + 2*x1) + 64*x3), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 4, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 16*((-2) + x2 + 2*x1) + 32*x3), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x4), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gc/cgcs3dutuobyyoqe3evm46dbg6gsn4kqpgimv5jhdscya6plo3en.py
# Topologically Sorted Source Nodes: [input_79], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_79 => add_59, mul_88, mul_89, sub_29
# Graph fragment:
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_237), kwargs = {})
#   %add_59 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_239), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4l/c4lqwzxbxusweqsutlotswapgsul6lopqfu7tj4xfnfdb6mfnt4s.py
# Topologically Sorted Source Nodes: [input_87], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_87 => add_65, mul_97, mul_98, sub_32
# Graph fragment:
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_257), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_261), kwargs = {})
#   %add_65 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_263), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/32/c32x4wyfl7jrhvsvvbledn7imtgahzhbeql4n56mgktilr4ubmqc.py
# Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_8 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_19, %relu_21], 1), kwargs = {})
triton_poi_fused_cat_17 = async_compile.triton('triton_poi_fused_cat_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 8*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 4, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 4*((-2) + x1) + 8*x2), tmp25 & xmask, other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-2) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ia/ciatcp2aovqxp6zly4ipp6r4zemg3ez3qxmcmmhzmnqrtcknnt5q.py
# Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_26 => clone_8
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_18 = async_compile.triton('triton_poi_fused_clone_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 2)
    x2 = ((xindex // 8) % 2)
    x3 = xindex // 16
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2 + 8*x1 + 16*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zg/czgffdkjw3wflyzg4b3yiynhudqax7fszokljqzkbmusp2b2axl5.py
# Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_92 => add_69, mul_103, mul_104, sub_34
#   input_93 => relu_22
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_34, %unsqueeze_273), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %unsqueeze_277), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_279), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_69,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/js/cjstuknj2wmm6kpipt46z2gw2532t2mguqkks5qez3zwwynrzrai.py
# Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_29 => clone_9
# Graph fragment:
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_20 = async_compile.triton('triton_poi_fused_clone_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 2)
    x2 = ((xindex // 8) % 2)
    x0 = (xindex % 4)
    x3 = xindex // 16
    x4 = xindex
    tmp0 = x2 + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x2 + 2*x1) + 16*x3), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 4, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4*((-2) + x2 + 2*x1) + 8*x3), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-2) + x2 + 2*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x4), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bx/cbxllheohecqd2vf6xzx72k45bxurb72zuctaqouw5kvzqqdxzqy.py
# Topologically Sorted Source Nodes: [input_116, input_117, v], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   input_116 => add_87, mul_130, mul_131, sub_43
#   input_117 => relu_28
#   v => mean
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_28 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_87,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_28, [-1, -2], True), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
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
    tmp19 = tmp18 - tmp1
    tmp20 = tmp19 * tmp10
    tmp21 = tmp20 * tmp12
    tmp22 = tmp21 + tmp14
    tmp23 = triton_helpers.maximum(tmp16, tmp22)
    tmp24 = tmp17 + tmp23
    tmp26 = tmp25 - tmp1
    tmp27 = tmp26 * tmp10
    tmp28 = tmp27 * tmp12
    tmp29 = tmp28 + tmp14
    tmp30 = triton_helpers.maximum(tmp16, tmp29)
    tmp31 = tmp24 + tmp30
    tmp33 = tmp32 - tmp1
    tmp34 = tmp33 * tmp10
    tmp35 = tmp34 * tmp12
    tmp36 = tmp35 + tmp14
    tmp37 = triton_helpers.maximum(tmp16, tmp36)
    tmp38 = tmp31 + tmp37
    tmp39 = 4.0
    tmp40 = tmp38 / tmp39
    tl.store(out_ptr0 + (x2), tmp40, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_13, (2, ), (1, ))
    assert_size_stride(primals_14, (2, ), (1, ))
    assert_size_stride(primals_15, (2, ), (1, ))
    assert_size_stride(primals_16, (2, ), (1, ))
    assert_size_stride(primals_17, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_18, (2, ), (1, ))
    assert_size_stride(primals_19, (2, ), (1, ))
    assert_size_stride(primals_20, (2, ), (1, ))
    assert_size_stride(primals_21, (2, ), (1, ))
    assert_size_stride(primals_22, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (2, ), (1, ))
    assert_size_stride(primals_24, (2, ), (1, ))
    assert_size_stride(primals_25, (2, ), (1, ))
    assert_size_stride(primals_26, (2, ), (1, ))
    assert_size_stride(primals_27, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_28, (2, ), (1, ))
    assert_size_stride(primals_29, (2, ), (1, ))
    assert_size_stride(primals_30, (2, ), (1, ))
    assert_size_stride(primals_31, (2, ), (1, ))
    assert_size_stride(primals_32, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_33, (2, ), (1, ))
    assert_size_stride(primals_34, (2, ), (1, ))
    assert_size_stride(primals_35, (2, ), (1, ))
    assert_size_stride(primals_36, (2, ), (1, ))
    assert_size_stride(primals_37, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_38, (2, ), (1, ))
    assert_size_stride(primals_39, (2, ), (1, ))
    assert_size_stride(primals_40, (2, ), (1, ))
    assert_size_stride(primals_41, (2, ), (1, ))
    assert_size_stride(primals_42, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_43, (2, ), (1, ))
    assert_size_stride(primals_44, (2, ), (1, ))
    assert_size_stride(primals_45, (2, ), (1, ))
    assert_size_stride(primals_46, (2, ), (1, ))
    assert_size_stride(primals_47, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_48, (2, ), (1, ))
    assert_size_stride(primals_49, (2, ), (1, ))
    assert_size_stride(primals_50, (2, ), (1, ))
    assert_size_stride(primals_51, (2, ), (1, ))
    assert_size_stride(primals_52, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_53, (2, ), (1, ))
    assert_size_stride(primals_54, (2, ), (1, ))
    assert_size_stride(primals_55, (2, ), (1, ))
    assert_size_stride(primals_56, (2, ), (1, ))
    assert_size_stride(primals_57, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_58, (2, ), (1, ))
    assert_size_stride(primals_59, (2, ), (1, ))
    assert_size_stride(primals_60, (2, ), (1, ))
    assert_size_stride(primals_61, (2, ), (1, ))
    assert_size_stride(primals_62, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_63, (2, ), (1, ))
    assert_size_stride(primals_64, (2, ), (1, ))
    assert_size_stride(primals_65, (2, ), (1, ))
    assert_size_stride(primals_66, (2, ), (1, ))
    assert_size_stride(primals_67, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (2, ), (1, ))
    assert_size_stride(primals_69, (2, ), (1, ))
    assert_size_stride(primals_70, (2, ), (1, ))
    assert_size_stride(primals_71, (2, ), (1, ))
    assert_size_stride(primals_72, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_73, (2, ), (1, ))
    assert_size_stride(primals_74, (2, ), (1, ))
    assert_size_stride(primals_75, (2, ), (1, ))
    assert_size_stride(primals_76, (2, ), (1, ))
    assert_size_stride(primals_77, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_78, (4, ), (1, ))
    assert_size_stride(primals_79, (4, ), (1, ))
    assert_size_stride(primals_80, (4, ), (1, ))
    assert_size_stride(primals_81, (4, ), (1, ))
    assert_size_stride(primals_82, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_83, (2, ), (1, ))
    assert_size_stride(primals_84, (2, ), (1, ))
    assert_size_stride(primals_85, (2, ), (1, ))
    assert_size_stride(primals_86, (2, ), (1, ))
    assert_size_stride(primals_87, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_88, (2, ), (1, ))
    assert_size_stride(primals_89, (2, ), (1, ))
    assert_size_stride(primals_90, (2, ), (1, ))
    assert_size_stride(primals_91, (2, ), (1, ))
    assert_size_stride(primals_92, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_93, (2, ), (1, ))
    assert_size_stride(primals_94, (2, ), (1, ))
    assert_size_stride(primals_95, (2, ), (1, ))
    assert_size_stride(primals_96, (2, ), (1, ))
    assert_size_stride(primals_97, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_98, (2, ), (1, ))
    assert_size_stride(primals_99, (2, ), (1, ))
    assert_size_stride(primals_100, (2, ), (1, ))
    assert_size_stride(primals_101, (2, ), (1, ))
    assert_size_stride(primals_102, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_103, (2, ), (1, ))
    assert_size_stride(primals_104, (2, ), (1, ))
    assert_size_stride(primals_105, (2, ), (1, ))
    assert_size_stride(primals_106, (2, ), (1, ))
    assert_size_stride(primals_107, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_108, (2, ), (1, ))
    assert_size_stride(primals_109, (2, ), (1, ))
    assert_size_stride(primals_110, (2, ), (1, ))
    assert_size_stride(primals_111, (2, ), (1, ))
    assert_size_stride(primals_112, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_113, (2, ), (1, ))
    assert_size_stride(primals_114, (2, ), (1, ))
    assert_size_stride(primals_115, (2, ), (1, ))
    assert_size_stride(primals_116, (2, ), (1, ))
    assert_size_stride(primals_117, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_118, (2, ), (1, ))
    assert_size_stride(primals_119, (2, ), (1, ))
    assert_size_stride(primals_120, (2, ), (1, ))
    assert_size_stride(primals_121, (2, ), (1, ))
    assert_size_stride(primals_122, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_123, (2, ), (1, ))
    assert_size_stride(primals_124, (2, ), (1, ))
    assert_size_stride(primals_125, (2, ), (1, ))
    assert_size_stride(primals_126, (2, ), (1, ))
    assert_size_stride(primals_127, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_128, (2, ), (1, ))
    assert_size_stride(primals_129, (2, ), (1, ))
    assert_size_stride(primals_130, (2, ), (1, ))
    assert_size_stride(primals_131, (2, ), (1, ))
    assert_size_stride(primals_132, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_133, (2, ), (1, ))
    assert_size_stride(primals_134, (2, ), (1, ))
    assert_size_stride(primals_135, (2, ), (1, ))
    assert_size_stride(primals_136, (2, ), (1, ))
    assert_size_stride(primals_137, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_138, (2, ), (1, ))
    assert_size_stride(primals_139, (2, ), (1, ))
    assert_size_stride(primals_140, (2, ), (1, ))
    assert_size_stride(primals_141, (2, ), (1, ))
    assert_size_stride(primals_142, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_143, (2, ), (1, ))
    assert_size_stride(primals_144, (2, ), (1, ))
    assert_size_stride(primals_145, (2, ), (1, ))
    assert_size_stride(primals_146, (2, ), (1, ))
    assert_size_stride(primals_147, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_148, (4, ), (1, ))
    assert_size_stride(primals_149, (4, ), (1, ))
    assert_size_stride(primals_150, (4, ), (1, ))
    assert_size_stride(primals_151, (4, ), (1, ))
    assert_size_stride(primals_152, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_153, (2, ), (1, ))
    assert_size_stride(primals_154, (2, ), (1, ))
    assert_size_stride(primals_155, (2, ), (1, ))
    assert_size_stride(primals_156, (2, ), (1, ))
    assert_size_stride(primals_157, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_158, (2, ), (1, ))
    assert_size_stride(primals_159, (2, ), (1, ))
    assert_size_stride(primals_160, (2, ), (1, ))
    assert_size_stride(primals_161, (2, ), (1, ))
    assert_size_stride(primals_162, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_163, (2, ), (1, ))
    assert_size_stride(primals_164, (2, ), (1, ))
    assert_size_stride(primals_165, (2, ), (1, ))
    assert_size_stride(primals_166, (2, ), (1, ))
    assert_size_stride(primals_167, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_168, (2, ), (1, ))
    assert_size_stride(primals_169, (2, ), (1, ))
    assert_size_stride(primals_170, (2, ), (1, ))
    assert_size_stride(primals_171, (2, ), (1, ))
    assert_size_stride(primals_172, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_173, (2, ), (1, ))
    assert_size_stride(primals_174, (2, ), (1, ))
    assert_size_stride(primals_175, (2, ), (1, ))
    assert_size_stride(primals_176, (2, ), (1, ))
    assert_size_stride(primals_177, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_178, (2, ), (1, ))
    assert_size_stride(primals_179, (2, ), (1, ))
    assert_size_stride(primals_180, (2, ), (1, ))
    assert_size_stride(primals_181, (2, ), (1, ))
    assert_size_stride(primals_182, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_183, (2, ), (1, ))
    assert_size_stride(primals_184, (2, ), (1, ))
    assert_size_stride(primals_185, (2, ), (1, ))
    assert_size_stride(primals_186, (2, ), (1, ))
    assert_size_stride(primals_187, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_188, (2, ), (1, ))
    assert_size_stride(primals_189, (2, ), (1, ))
    assert_size_stride(primals_190, (2, ), (1, ))
    assert_size_stride(primals_191, (2, ), (1, ))
    assert_size_stride(primals_192, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (2, ), (1, ))
    assert_size_stride(primals_194, (2, ), (1, ))
    assert_size_stride(primals_195, (2, ), (1, ))
    assert_size_stride(primals_196, (2, ), (1, ))
    assert_size_stride(primals_197, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_198, (2, ), (1, ))
    assert_size_stride(primals_199, (2, ), (1, ))
    assert_size_stride(primals_200, (2, ), (1, ))
    assert_size_stride(primals_201, (2, ), (1, ))
    assert_size_stride(primals_202, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_203, (2, ), (1, ))
    assert_size_stride(primals_204, (2, ), (1, ))
    assert_size_stride(primals_205, (2, ), (1, ))
    assert_size_stride(primals_206, (2, ), (1, ))
    assert_size_stride(primals_207, (2, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_208, (2, ), (1, ))
    assert_size_stride(primals_209, (2, ), (1, ))
    assert_size_stride(primals_210, (2, ), (1, ))
    assert_size_stride(primals_211, (2, ), (1, ))
    assert_size_stride(primals_212, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_213, (2, ), (1, ))
    assert_size_stride(primals_214, (2, ), (1, ))
    assert_size_stride(primals_215, (2, ), (1, ))
    assert_size_stride(primals_216, (2, ), (1, ))
    assert_size_stride(primals_217, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_218, (4, ), (1, ))
    assert_size_stride(primals_219, (4, ), (1, ))
    assert_size_stride(primals_220, (4, ), (1, ))
    assert_size_stride(primals_221, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 16384, grid=grid(16384), stream=stream0)
        del primals_6
        buf2 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(buf1, buf2, buf3, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, primals_7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf4, (4, 4, 8, 8), (256, 64, 8, 1))
        buf5 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf4, primals_8, primals_9, primals_10, primals_11, buf5, 1024, grid=grid(1024), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 2, 8, 8), (128, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf2, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 2, 16, 16), (512, 256, 16, 1))
        buf8 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf7, primals_18, primals_19, primals_20, primals_21, buf8, 2048, grid=grid(2048), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf9, (4, 2, 8, 8), (128, 64, 8, 1))
        buf10 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf9, primals_23, primals_24, primals_25, primals_26, buf10, 512, grid=grid(512), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 2, 8, 8), (128, 64, 8, 1))
        buf12 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf6, primals_13, primals_14, primals_15, primals_16, buf11, primals_28, primals_29, primals_30, primals_31, buf12, 1024, grid=grid(1024), stream=stream0)
        buf13 = empty_strided_cuda((4, 2, 2, 8, 8), (256, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf12, buf13, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(reinterpret_tensor(buf13, (4, 2, 8, 8), (256, 64, 8, 1), 128), primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 2, 8, 8), (128, 64, 8, 1))
        buf15 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf14, primals_33, primals_34, primals_35, primals_36, buf15, 512, grid=grid(512), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf16, (4, 2, 8, 8), (128, 64, 8, 1))
        buf17 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf16, primals_38, primals_39, primals_40, primals_41, buf17, 512, grid=grid(512), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 2, 8, 8), (128, 64, 8, 1))
        buf19 = reinterpret_tensor(buf12, (4, 2, 2, 8, 8), (256, 128, 64, 8, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf13, buf18, primals_43, primals_44, primals_45, primals_46, buf19, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(reinterpret_tensor(buf19, (4, 2, 8, 8), (256, 64, 8, 1), 128), primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 2, 8, 8), (128, 64, 8, 1))
        buf21 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf20, primals_48, primals_49, primals_50, primals_51, buf21, 512, grid=grid(512), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf22, (4, 2, 8, 8), (128, 64, 8, 1))
        buf23 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf22, primals_53, primals_54, primals_55, primals_56, buf23, 512, grid=grid(512), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 2, 8, 8), (128, 64, 8, 1))
        buf25 = empty_strided_cuda((4, 2, 2, 8, 8), (256, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf19, buf24, primals_58, primals_59, primals_60, primals_61, buf25, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(reinterpret_tensor(buf25, (4, 2, 8, 8), (256, 64, 8, 1), 128), primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 2, 8, 8), (128, 64, 8, 1))
        buf27 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf26, primals_63, primals_64, primals_65, primals_66, buf27, 512, grid=grid(512), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf28, (4, 2, 8, 8), (128, 64, 8, 1))
        buf29 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf28, primals_68, primals_69, primals_70, primals_71, buf29, 512, grid=grid(512), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 2, 8, 8), (128, 64, 8, 1))
        buf31 = empty_strided_cuda((4, 2, 2, 8, 8), (256, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf25, buf30, primals_73, primals_74, primals_75, primals_76, buf31, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(reinterpret_tensor(buf31, (4, 4, 8, 8), (256, 64, 8, 1), 0), primals_77, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf32, (4, 4, 4, 4), (64, 16, 4, 1))
        buf33 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf32, primals_78, primals_79, primals_80, primals_81, buf33, 256, grid=grid(256), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 2, 4, 4), (32, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(reinterpret_tensor(buf31, (4, 4, 8, 8), (256, 64, 8, 1), 0), primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 2, 8, 8), (128, 64, 8, 1))
        buf36 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_47, input_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf35, primals_88, primals_89, primals_90, primals_91, buf36, 512, grid=grid(512), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_92, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf37, (4, 2, 4, 4), (32, 16, 4, 1))
        buf38 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf37, primals_93, primals_94, primals_95, primals_96, buf38, 128, grid=grid(128), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 2, 4, 4), (32, 16, 4, 1))
        buf40 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf34, primals_83, primals_84, primals_85, primals_86, buf39, primals_98, primals_99, primals_100, primals_101, buf40, 256, grid=grid(256), stream=stream0)
        buf41 = empty_strided_cuda((4, 2, 2, 4, 4), (64, 32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf40, buf41, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(reinterpret_tensor(buf41, (4, 2, 4, 4), (64, 16, 4, 1), 32), primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 2, 4, 4), (32, 16, 4, 1))
        buf43 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_55, input_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf42, primals_103, primals_104, primals_105, primals_106, buf43, 128, grid=grid(128), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf44, (4, 2, 4, 4), (32, 16, 4, 1))
        buf45 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf44, primals_108, primals_109, primals_110, primals_111, buf45, 128, grid=grid(128), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 2, 4, 4), (32, 16, 4, 1))
        buf47 = reinterpret_tensor(buf40, (4, 2, 2, 4, 4), (64, 32, 16, 4, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_14.run(buf41, buf46, primals_113, primals_114, primals_115, primals_116, buf47, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(reinterpret_tensor(buf47, (4, 2, 4, 4), (64, 16, 4, 1), 32), primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 2, 4, 4), (32, 16, 4, 1))
        buf49 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_63, input_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf48, primals_118, primals_119, primals_120, primals_121, buf49, 128, grid=grid(128), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf50, (4, 2, 4, 4), (32, 16, 4, 1))
        buf51 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf50, primals_123, primals_124, primals_125, primals_126, buf51, 128, grid=grid(128), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 2, 4, 4), (32, 16, 4, 1))
        buf53 = empty_strided_cuda((4, 2, 2, 4, 4), (64, 32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_14.run(buf47, buf52, primals_128, primals_129, primals_130, primals_131, buf53, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(reinterpret_tensor(buf53, (4, 2, 4, 4), (64, 16, 4, 1), 32), primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 2, 4, 4), (32, 16, 4, 1))
        buf55 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_71, input_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf54, primals_133, primals_134, primals_135, primals_136, buf55, 128, grid=grid(128), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf56, (4, 2, 4, 4), (32, 16, 4, 1))
        buf57 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf56, primals_138, primals_139, primals_140, primals_141, buf57, 128, grid=grid(128), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 2, 4, 4), (32, 16, 4, 1))
        buf59 = empty_strided_cuda((4, 2, 2, 4, 4), (64, 32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_14.run(buf53, buf58, primals_143, primals_144, primals_145, primals_146, buf59, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(reinterpret_tensor(buf59, (4, 4, 4, 4), (64, 16, 4, 1), 0), primals_147, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf60, (4, 4, 2, 2), (16, 4, 2, 1))
        buf61 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf60, primals_148, primals_149, primals_150, primals_151, buf61, 64, grid=grid(64), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 2, 2, 2), (8, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(reinterpret_tensor(buf59, (4, 4, 4, 4), (64, 16, 4, 1), 0), primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 2, 4, 4), (32, 16, 4, 1))
        buf64 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_84, input_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf63, primals_158, primals_159, primals_160, primals_161, buf64, 128, grid=grid(128), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_162, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf65, (4, 2, 2, 2), (8, 4, 2, 1))
        buf66 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf65, primals_163, primals_164, primals_165, primals_166, buf66, 32, grid=grid(32), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 2, 2, 2), (8, 4, 2, 1))
        buf68 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_17.run(buf62, primals_153, primals_154, primals_155, primals_156, buf67, primals_168, primals_169, primals_170, primals_171, buf68, 64, grid=grid(64), stream=stream0)
        buf69 = empty_strided_cuda((4, 2, 2, 2, 2), (16, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_18.run(buf68, buf69, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(reinterpret_tensor(buf69, (4, 2, 2, 2), (16, 4, 2, 1), 8), primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 2, 2, 2), (8, 4, 2, 1))
        buf71 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf70, primals_173, primals_174, primals_175, primals_176, buf71, 32, grid=grid(32), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf72, (4, 2, 2, 2), (8, 4, 2, 1))
        buf73 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf72, primals_178, primals_179, primals_180, primals_181, buf73, 32, grid=grid(32), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 2, 2, 2), (8, 4, 2, 1))
        buf75 = reinterpret_tensor(buf68, (4, 2, 2, 2, 2), (16, 8, 4, 2, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_20.run(buf69, buf74, primals_183, primals_184, primals_185, primals_186, buf75, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(reinterpret_tensor(buf75, (4, 2, 2, 2), (16, 4, 2, 1), 8), primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 2, 2, 2), (8, 4, 2, 1))
        buf77 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_100, input_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf76, primals_188, primals_189, primals_190, primals_191, buf77, 32, grid=grid(32), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf78, (4, 2, 2, 2), (8, 4, 2, 1))
        buf79 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf78, primals_193, primals_194, primals_195, primals_196, buf79, 32, grid=grid(32), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 2, 2, 2), (8, 4, 2, 1))
        buf81 = empty_strided_cuda((4, 2, 2, 2, 2), (16, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_20.run(buf75, buf80, primals_198, primals_199, primals_200, primals_201, buf81, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_107], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(reinterpret_tensor(buf81, (4, 2, 2, 2), (16, 4, 2, 1), 8), primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 2, 2, 2), (8, 4, 2, 1))
        buf83 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_108, input_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf82, primals_203, primals_204, primals_205, primals_206, buf83, 32, grid=grid(32), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf84, (4, 2, 2, 2), (8, 4, 2, 1))
        buf85 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_111], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf84, primals_208, primals_209, primals_210, primals_211, buf85, 32, grid=grid(32), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 2, 2, 2), (8, 4, 2, 1))
        buf87 = empty_strided_cuda((4, 2, 2, 2, 2), (16, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_20.run(buf81, buf86, primals_213, primals_214, primals_215, primals_216, buf87, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(reinterpret_tensor(buf87, (4, 4, 2, 2), (16, 4, 2, 1), 0), primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 4, 2, 2), (16, 4, 2, 1))
        buf89 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_116, input_117, v], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_21.run(buf88, primals_218, primals_219, primals_220, primals_221, buf89, 16, grid=grid(16), stream=stream0)
    return (reinterpret_tensor(buf89, (4, 4), (4, 1), 0), primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, reinterpret_tensor(buf13, (4, 2, 8, 8), (256, 64, 8, 1), 128), buf14, buf15, buf16, buf17, buf18, reinterpret_tensor(buf19, (4, 2, 8, 8), (256, 64, 8, 1), 128), buf20, buf21, buf22, buf23, buf24, reinterpret_tensor(buf25, (4, 2, 8, 8), (256, 64, 8, 1), 128), buf26, buf27, buf28, buf29, buf30, reinterpret_tensor(buf31, (4, 4, 8, 8), (256, 64, 8, 1), 0), buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, reinterpret_tensor(buf41, (4, 2, 4, 4), (64, 16, 4, 1), 32), buf42, buf43, buf44, buf45, buf46, reinterpret_tensor(buf47, (4, 2, 4, 4), (64, 16, 4, 1), 32), buf48, buf49, buf50, buf51, buf52, reinterpret_tensor(buf53, (4, 2, 4, 4), (64, 16, 4, 1), 32), buf54, buf55, buf56, buf57, buf58, reinterpret_tensor(buf59, (4, 4, 4, 4), (64, 16, 4, 1), 0), buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, reinterpret_tensor(buf69, (4, 2, 2, 2), (16, 4, 2, 1), 8), buf70, buf71, buf72, buf73, buf74, reinterpret_tensor(buf75, (4, 2, 2, 2), (16, 4, 2, 1), 8), buf76, buf77, buf78, buf79, buf80, reinterpret_tensor(buf81, (4, 2, 2, 2), (16, 4, 2, 1), 8), buf82, buf83, buf84, buf85, buf86, reinterpret_tensor(buf87, (4, 4, 2, 2), (16, 4, 2, 1), 0), buf88, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((2, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
