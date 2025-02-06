# AOT ID: ['6_forward']
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


# kernel path: inductor_cache/fi/cficfwunusrpkrq34t2fqm2e7nqrpfhwb7ocd6jtycjc4ei2jdz7.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    x1 = ((xindex // 4096) % 16)
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


# kernel path: inductor_cache/5a/c5a7xdzuu47p6z53sporaqnlg2azil7gmps6bxvr6y2grpgv6t5l.py
# Topologically Sorted Source Nodes: [maxpool0], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   maxpool0 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/dd/cdd2luabwlku737gxy2abzzcnddgktmr5frh67kyenen6ygfwxlo.py
# Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_7 => convolution_2
#   input_8 => add_5, mul_7, mul_8, sub_2
#   input_9 => relu_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_14, %primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
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


# kernel path: inductor_cache/l2/cl24z6waopzxpf36qejympec5pm32tod47ep3qeggiykfwxilud3.py
# Topologically Sorted Source Nodes: [maxpool1], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   maxpool1 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_3 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/gp/cgpavxglkpa7aem3etw24idwfoqrwnuqm7d3vn3g5kdzfwtvdyse.py
# Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_13 => convolution_4
#   input_14 => add_9, mul_13, mul_14, sub_4
#   input_15 => relu_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_26, %primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ya/cya7vgvhtxhlx77hrt7sx3y4dbh4n4ahc4crgmnxgkdv6vswxzux.py
# Topologically Sorted Source Nodes: [maxpool2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   maxpool2 => getitem_4, getitem_5
# Graph fragment:
#   %getitem_4 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_5 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_5(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/z6/cz6tk2bmobbymdpyd6tduopy3zmcdj4gjhblgjyitr3rq5ka3ioq.py
# Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_19 => convolution_6
#   input_20 => add_13, mul_19, mul_20, sub_6
#   input_21 => relu_6
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_38, %primals_39, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/36/c36b2zl7ag7rq4lylsm35yuzwejtmbknhlvbczlulw44cytnmcdl.py
# Topologically Sorted Source Nodes: [maxpool3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   maxpool3 => getitem_6, getitem_7
# Graph fragment:
#   %getitem_6 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 0), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_7 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/2k/c2kxtaxlb2tuox7tmmzkxihcxbs7ti6qt6yvxfibw3kdj4t6hbbj.py
# Topologically Sorted Source Nodes: [input_25, input_26, input_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_25 => convolution_8
#   input_26 => add_17, mul_25, mul_26, sub_8
#   input_27 => relu_8
# Graph fragment:
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %primals_50, %primals_51, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/4p/c4peziognz3fck5hdozwx6zexku4w2stxrunnepakikccvfegw5m.py
# Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_32 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %primals_62, %primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 3)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w3/cw3zx7zi6w5f5epgqwhw7te4thcdki3qu3zszbxsolvjdbudnm52.py
# Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.adaptive_max_pool2d, aten.sigmoid, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   input_33 => _low_memory_max_pool2d_with_offsets_4, getitem_9
#   input_34 => sigmoid
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%convolution_10, [4, 4], [4, 4], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_9 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_4, 1), kwargs = {})
#   %sigmoid : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem_8,), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub_13), kwargs = {})
triton_poi_fused_adaptive_max_pool2d_sigmoid_sigmoid_backward_10 = async_compile.triton('triton_poi_fused_adaptive_max_pool2d_sigmoid_sigmoid_backward_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_adaptive_max_pool2d_sigmoid_sigmoid_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_adaptive_max_pool2d_sigmoid_sigmoid_backward_10(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tmp31 = tmp1 > tmp0
    tmp32 = tl.full([1], 1, tl.int8)
    tmp33 = tl.full([1], 0, tl.int8)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tmp3 > tmp2
    tmp36 = tl.full([1], 2, tl.int8)
    tmp37 = tl.where(tmp35, tmp36, tmp34)
    tmp38 = tmp5 > tmp4
    tmp39 = tl.full([1], 3, tl.int8)
    tmp40 = tl.where(tmp38, tmp39, tmp37)
    tmp41 = tmp7 > tmp6
    tmp42 = tl.full([1], 4, tl.int8)
    tmp43 = tl.where(tmp41, tmp42, tmp40)
    tmp44 = tmp9 > tmp8
    tmp45 = tl.full([1], 5, tl.int8)
    tmp46 = tl.where(tmp44, tmp45, tmp43)
    tmp47 = tmp11 > tmp10
    tmp48 = tl.full([1], 6, tl.int8)
    tmp49 = tl.where(tmp47, tmp48, tmp46)
    tmp50 = tmp13 > tmp12
    tmp51 = tl.full([1], 7, tl.int8)
    tmp52 = tl.where(tmp50, tmp51, tmp49)
    tmp53 = tmp15 > tmp14
    tmp54 = tl.full([1], 8, tl.int8)
    tmp55 = tl.where(tmp53, tmp54, tmp52)
    tmp56 = tmp17 > tmp16
    tmp57 = tl.full([1], 9, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp19 > tmp18
    tmp60 = tl.full([1], 10, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp21 > tmp20
    tmp63 = tl.full([1], 11, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp23 > tmp22
    tmp66 = tl.full([1], 12, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp25 > tmp24
    tmp69 = tl.full([1], 13, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp27 > tmp26
    tmp72 = tl.full([1], 14, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp29 > tmp28
    tmp75 = tl.full([1], 15, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tl.sigmoid(tmp30)
    tmp78 = 1.0
    tmp79 = tmp78 - tmp77
    tmp80 = tmp77 * tmp79
    tl.store(out_ptr0 + (x0), tmp76, xmask)
    tl.store(in_out_ptr0 + (x0), tmp77, xmask)
    tl.store(out_ptr1 + (x0), tmp80, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m7/cm7blb5fmpun3nyammtwi3hon3c7nzo2zkrucxomph5yvcqbbfvz.py
# Topologically Sorted Source Nodes: [outputs0_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs0_1 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_11, %relu_1], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 32)
    x0 = (xindex % 4096)
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 65536*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 32, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (x0 + 4096*((-16) + x1) + 65536*x2), tmp10, other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/kz/ckzitlagwchgnk7lita56rc7uxeq3vjylx6sg6lmibyrlhcgsmni.py
# Topologically Sorted Source Nodes: [input_35, input_36], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_35 => convolution_12
#   input_36 => relu_10
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_66, %primals_67, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
triton_poi_fused_convolution_relu_12 = async_compile.triton('triton_poi_fused_convolution_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/qz/cqz7nccp44tehptuybtgfa6ib2tumtg2hcrjbharbdzg23l5lfzk.py
# Topologically Sorted Source Nodes: [outputs0_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs0_3 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_14, %relu_3], 1), kwargs = {})
triton_poi_fused_cat_13 = async_compile.triton('triton_poi_fused_cat_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 64)
    x0 = (xindex % 1024)
    x2 = xindex // 65536
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 32768*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 64, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (x0 + 1024*((-32) + x1) + 32768*x2), tmp10, other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/sk/cskj74zet6x4tvqbmd5eqtfagqv2d7mqwoxwqxt5ox5t5pvbtmei.py
# Topologically Sorted Source Nodes: [input_39, input_40], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_39 => convolution_15
#   input_40 => relu_12
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %primals_72, %primals_73, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
triton_poi_fused_convolution_relu_14 = async_compile.triton('triton_poi_fused_convolution_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/6d/c6dwbc6wuo7gohwyaaidnq6hfhjh44xsaxvrttzgwaxxytyvc4hl.py
# Topologically Sorted Source Nodes: [outputs0_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs0_5 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_17, %relu_5], 1), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_poi_fused_cat_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 128)
    x0 = (xindex % 256)
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 16384*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 128, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (x0 + 256*((-64) + x1) + 16384*x2), tmp10, other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/eb/cebxyvi4gxu5byugu5eedluqyetsvstuaeflh642yz7ogiu67zpc.py
# Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_43 => convolution_18
#   input_44 => relu_14
# Graph fragment:
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %primals_78, %primals_79, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
triton_poi_fused_convolution_relu_16 = async_compile.triton('triton_poi_fused_convolution_relu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/6m/c6mdoocxe2q5x3bvrw6y4m2uck4veltlxwpx2aagfx2pl6mzrqn2.py
# Topologically Sorted Source Nodes: [outputs0_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs0_7 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_20, %relu_7], 1), kwargs = {})
triton_poi_fused_cat_17 = async_compile.triton('triton_poi_fused_cat_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 256)
    x0 = (xindex % 64)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 256, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (x0 + 64*((-128) + x1) + 8192*x2), tmp10, other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/43/c43bv2eeqzwwkwy7bxlhuku4beultuil7qezclg4blis37yomuts.py
# Topologically Sorted Source Nodes: [input_47, input_48], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_47 => convolution_21
#   input_48 => relu_16
# Graph fragment:
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %primals_84, %primals_85, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
triton_poi_fused_convolution_relu_18 = async_compile.triton('triton_poi_fused_convolution_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/as/cas46bw5ug3yney2ulgewes4jn4zvosbt2vcgoxigd2j2jcwnsmg.py
# Topologically Sorted Source Nodes: [outputs0_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs0_10 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_4, %relu_11], 1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 48)
    x0 = (xindex % 4096)
    x2 = xindex // 196608
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 16, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 65536*x2), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tmp5 >= tmp8
    tmp17 = tl.full([1], 32, tl.int64)
    tmp18 = tmp5 < tmp17
    tmp19 = tmp16 & tmp4
    tmp20 = tl.load(in_ptr2 + (x0 + 4096*((-16) + (x1)) + 65536*x2), tmp19, other=0.0)
    tmp21 = tl.where(tmp9, tmp15, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 48, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (x0 + 4096*((-32) + x1) + 65536*x2), tmp24, other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/wy/cwyrxefembaika5j775ofqx2tcywrmjer4ukfmkm4mvlrddi3y3r.py
# Topologically Sorted Source Nodes: [outputs0_13], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs0_13 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_6, %relu_13], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 96)
    x0 = (xindex % 1024)
    x2 = xindex // 98304
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 32, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 32768*x2), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tmp5 >= tmp8
    tmp17 = tl.full([1], 64, tl.int64)
    tmp18 = tmp5 < tmp17
    tmp19 = tmp16 & tmp4
    tmp20 = tl.load(in_ptr2 + (x0 + 1024*((-32) + (x1)) + 32768*x2), tmp19, other=0.0)
    tmp21 = tl.where(tmp9, tmp15, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 96, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (x0 + 1024*((-64) + x1) + 32768*x2), tmp24, other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/tz/ctzmj3cdokserix4xzxm42ijvzzu4jduxghvyu2dft6hug2g6rx3.py
# Topologically Sorted Source Nodes: [outputs0_16], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs0_16 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_8, %relu_15], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 192)
    x0 = (xindex % 256)
    x2 = xindex // 49152
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 64, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x0 + 256*(x1) + 16384*x2), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tmp5 >= tmp8
    tmp17 = tl.full([1], 128, tl.int64)
    tmp18 = tmp5 < tmp17
    tmp19 = tmp16 & tmp4
    tmp20 = tl.load(in_ptr2 + (x0 + 256*((-64) + (x1)) + 16384*x2), tmp19, other=0.0)
    tmp21 = tl.where(tmp9, tmp15, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 192, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (x0 + 256*((-128) + x1) + 16384*x2), tmp24, other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/hk/chked3yds32xv6dpb3a5rpzrkjhxe7phcebpvxzzaaorfyub4llg.py
# Topologically Sorted Source Nodes: [outputs0_24], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs0_24 => cat_15
# Graph fragment:
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_14, %relu_21], 1), kwargs = {})
triton_poi_fused_cat_22 = async_compile.triton('triton_poi_fused_cat_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 128)
    x0 = (xindex % 1024)
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 64, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 32, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 32768*x2), tmp16, other=0.0)
    tmp18 = tl.load(in_ptr1 + (x1), tmp16, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tmp11 >= tmp14
    tmp23 = tl.full([1], 64, tl.int64)
    tmp24 = tmp11 < tmp23
    tmp25 = tmp22 & tmp10
    tmp26 = tl.load(in_ptr2 + (x0 + 1024*((-32) + (x1)) + 32768*x2), tmp25, other=0.0)
    tmp27 = tl.where(tmp15, tmp21, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp10, tmp27, tmp28)
    tmp30 = tmp5 >= tmp8
    tmp31 = tl.full([1], 96, tl.int64)
    tmp32 = tmp5 < tmp31
    tmp33 = tmp30 & tmp4
    tmp34 = tl.load(in_ptr3 + (x0 + 1024*((-64) + (x1)) + 32768*x2), tmp33, other=0.0)
    tmp35 = tl.where(tmp9, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp4, tmp35, tmp36)
    tmp38 = tmp0 >= tmp3
    tmp39 = tl.full([1], 128, tl.int64)
    tmp40 = tmp0 < tmp39
    tmp41 = tl.load(in_ptr4 + (x0 + 1024*((-96) + x1) + 32768*x2), tmp38, other=0.0)
    tmp42 = tl.where(tmp4, tmp37, tmp41)
    tl.store(out_ptr0 + (x3), tmp42, None)
''', device_str='cuda')


# kernel path: inductor_cache/yc/cycw2q5vkje4ra2irdlpqc72iiqtsmm3fou2l6sig5snucjumup3.py
# Topologically Sorted Source Nodes: [outputs0_20, outputs0_28], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs0_20 => cat_12
#   outputs0_28 => cat_18
# Graph fragment:
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_11, %relu_19], 1), kwargs = {})
#   %cat_18 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_17, %relu_19], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 64)
    x0 = (xindex % 4096)
    x2 = xindex // 262144
    x3 = xindex
    x4 = (xindex % 262144)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 48, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 32, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 16, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 65536*x2), tmp16, other=0.0)
    tmp18 = tl.load(in_ptr1 + (x1), tmp16, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tmp11 >= tmp14
    tmp23 = tl.full([1], 32, tl.int64)
    tmp24 = tmp11 < tmp23
    tmp25 = tmp22 & tmp10
    tmp26 = tl.load(in_ptr2 + (x0 + 4096*((-16) + (x1)) + 65536*x2), tmp25, other=0.0)
    tmp27 = tl.where(tmp15, tmp21, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp10, tmp27, tmp28)
    tmp30 = tmp5 >= tmp8
    tmp31 = tl.full([1], 48, tl.int64)
    tmp32 = tmp5 < tmp31
    tmp33 = tmp30 & tmp4
    tmp34 = tl.load(in_ptr3 + (x0 + 4096*((-32) + (x1)) + 65536*x2), tmp33, other=0.0)
    tmp35 = tl.where(tmp9, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp4, tmp35, tmp36)
    tmp38 = tmp0 >= tmp3
    tmp39 = tl.full([1], 64, tl.int64)
    tmp40 = tmp0 < tmp39
    tmp41 = tl.load(in_ptr4 + (x0 + 4096*((-48) + x1) + 65536*x2), tmp38, other=0.0)
    tmp42 = tl.where(tmp4, tmp37, tmp41)
    tmp43 = tl.load(in_ptr5 + (x0 + 4096*(x1) + 65536*x2), tmp16, other=0.0)
    tmp44 = tl.load(in_ptr6 + (x1), tmp16, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp16, tmp45, tmp46)
    tmp48 = tl.where(tmp15, tmp47, tmp26)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp10, tmp48, tmp49)
    tmp51 = tl.where(tmp9, tmp50, tmp34)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp4, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp53, tmp41)
    tl.store(out_ptr0 + (x3), tmp42, None)
    tl.store(out_ptr1 + (x4 + 327680*x2), tmp54, None)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdx5jl3e7gyb27wdhrhjzwn4s2gpazdynwnclbdh4hebgf6iq25f.py
# Topologically Sorted Source Nodes: [input_65, input_66, outputs0_29], Original ATen: [aten.convolution, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   input_65 => convolution_34
#   input_66 => relu_25
#   outputs0_29 => cat_19
# Graph fragment:
#   %convolution_34 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %primals_110, %primals_111, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_34,), kwargs = {})
#   %cat_19 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_18, %relu_25], 1), kwargs = {})
triton_poi_fused_cat_convolution_relu_24 = async_compile.triton('triton_poi_fused_cat_convolution_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_24(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 4096) % 16)
    x2 = xindex // 65536
    x3 = (xindex % 65536)
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x4), tmp4, None)
    tl.store(out_ptr0 + (x3 + 327680*x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/q2/cq2waez6bssjpy62ofxqeojjadwn7fm4hva6s2q6lvmgwmz2yy6d.py
# Topologically Sorted Source Nodes: [final_1, final_2, final_3, final_4, add, add_1, add_2], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add => add_20
#   add_1 => add_21
#   add_2 => add_22
#   final_1 => convolution_41
#   final_2 => convolution_42
#   final_3 => convolution_43
#   final_4 => convolution_44
# Graph fragment:
#   %convolution_41 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_124, %primals_125, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_42 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %primals_126, %primals_127, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_43 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %primals_128, %primals_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_29, %primals_130, %primals_131, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_41, %convolution_42), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %convolution_43), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %convolution_44), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 1), kwargs = {})
triton_poi_fused_add_convolution_25 = async_compile.triton('triton_poi_fused_add_convolution_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x3), None)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x3), None)
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/7a/c7a72t57r6fbao6xqxf2gwb4gum3sbsruydxbkc67guxbnof6vsy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 4), kwargs = {})
triton_poi_fused_26 = async_compile.triton('triton_poi_fused_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 4096)
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384*x2), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0 + 16384*x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0 + 16384*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12288 + x0 + 16384*x2), None, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = 0.25
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/al/caldbrpwy4g6nm3v4pggdnj7bdjfjj2pbpmvvoh6q6qp44qrerye.py
# Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => exp, log, sub_11, sum_1
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_tensor, %log), kwargs = {})
triton_poi_fused__log_softmax_27 = async_compile.triton('triton_poi_fused__log_softmax_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 4096)
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384*x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 16384*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (8192 + x0 + 16384*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (12288 + x0 + 16384*x2), None, eviction_policy='evict_last')
    tmp2 = tl_math.exp(tmp1)
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp2 + tmp4
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp5 + tmp7
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = tl_math.log(tmp11)
    tmp13 = tmp0 - tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (16, ), (1, ))
    assert_size_stride(primals_12, (16, ), (1, ))
    assert_size_stride(primals_13, (16, ), (1, ))
    assert_size_stride(primals_14, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (32, ), (1, ))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (256, ), (1, ))
    assert_size_stride(primals_56, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (3, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (3, ), (1, ))
    assert_size_stride(primals_64, (32, 16, 4, 4), (256, 16, 4, 1))
    assert_size_stride(primals_65, (16, ), (1, ))
    assert_size_stride(primals_66, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_67, (16, ), (1, ))
    assert_size_stride(primals_68, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_69, (16, ), (1, ))
    assert_size_stride(primals_70, (64, 32, 4, 4), (512, 16, 4, 1))
    assert_size_stride(primals_71, (32, ), (1, ))
    assert_size_stride(primals_72, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_73, (32, ), (1, ))
    assert_size_stride(primals_74, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_75, (32, ), (1, ))
    assert_size_stride(primals_76, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (256, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (32, 16, 4, 4), (256, 16, 4, 1))
    assert_size_stride(primals_89, (16, ), (1, ))
    assert_size_stride(primals_90, (16, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_91, (16, ), (1, ))
    assert_size_stride(primals_92, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_93, (16, ), (1, ))
    assert_size_stride(primals_94, (64, 32, 4, 4), (512, 16, 4, 1))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_97, (32, ), (1, ))
    assert_size_stride(primals_98, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_103, (64, ), (1, ))
    assert_size_stride(primals_104, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_105, (64, ), (1, ))
    assert_size_stride(primals_106, (32, 16, 4, 4), (256, 16, 4, 1))
    assert_size_stride(primals_107, (16, ), (1, ))
    assert_size_stride(primals_108, (16, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_109, (16, ), (1, ))
    assert_size_stride(primals_110, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_111, (16, ), (1, ))
    assert_size_stride(primals_112, (64, 32, 4, 4), (512, 16, 4, 1))
    assert_size_stride(primals_113, (32, ), (1, ))
    assert_size_stride(primals_114, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_115, (32, ), (1, ))
    assert_size_stride(primals_116, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_117, (32, ), (1, ))
    assert_size_stride(primals_118, (32, 16, 4, 4), (256, 16, 4, 1))
    assert_size_stride(primals_119, (16, ), (1, ))
    assert_size_stride(primals_120, (16, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_121, (16, ), (1, ))
    assert_size_stride(primals_122, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_123, (16, ), (1, ))
    assert_size_stride(primals_124, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_125, (4, ), (1, ))
    assert_size_stride(primals_126, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_127, (4, ), (1, ))
    assert_size_stride(primals_128, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_129, (4, ), (1, ))
    assert_size_stride(primals_130, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_131, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, primals_3, primals_4, primals_5, primals_6, primals_7, buf2, 262144, grid=grid(262144), stream=stream0)
        del primals_3
        del primals_7
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf4, primals_9, primals_10, primals_11, primals_12, primals_13, buf5, 262144, grid=grid(262144), stream=stream0)
        del primals_13
        del primals_9
        buf6 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [maxpool0], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(buf5, buf6, buf7, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf6, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf9, primals_15, primals_16, primals_17, primals_18, primals_19, buf10, 131072, grid=grid(131072), stream=stream0)
        del primals_15
        del primals_19
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf12 = buf11; del buf11  # reuse
        buf13 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf12, primals_21, primals_22, primals_23, primals_24, primals_25, buf13, 131072, grid=grid(131072), stream=stream0)
        del primals_21
        del primals_25
        buf14 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf15 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [maxpool1], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_3.run(buf13, buf14, buf15, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf14, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf17, primals_27, primals_28, primals_29, primals_30, primals_31, buf18, 65536, grid=grid(65536), stream=stream0)
        del primals_27
        del primals_31
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf20 = buf19; del buf19  # reuse
        buf21 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf20, primals_33, primals_34, primals_35, primals_36, primals_37, buf21, 65536, grid=grid(65536), stream=stream0)
        del primals_33
        del primals_37
        buf22 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf23 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [maxpool2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_5.run(buf21, buf22, buf23, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf22, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf25 = buf24; del buf24  # reuse
        buf26 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf25, primals_39, primals_40, primals_41, primals_42, primals_43, buf26, 32768, grid=grid(32768), stream=stream0)
        del primals_39
        del primals_43
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf28 = buf27; del buf27  # reuse
        buf29 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf28, primals_45, primals_46, primals_47, primals_48, primals_49, buf29, 32768, grid=grid(32768), stream=stream0)
        del primals_45
        del primals_49
        buf30 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        buf31 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [maxpool3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_7.run(buf29, buf30, buf31, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf30, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_25, input_26, input_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf33, primals_51, primals_52, primals_53, primals_54, primals_55, buf34, 16384, grid=grid(16384), stream=stream0)
        del primals_51
        del primals_55
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, input_29, input_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf36, primals_57, primals_58, primals_59, primals_60, primals_61, buf37, 16384, grid=grid(16384), stream=stream0)
        del primals_57
        del primals_61
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 3, 4, 4), (48, 16, 4, 1))
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf39, primals_63, 192, grid=grid(192), stream=stream0)
        del primals_63
        buf40 = empty_strided_cuda((4, 3, 1, 1), (3, 1, 12, 12), torch.float32)
        buf41 = empty_strided_cuda((4, 3, 1, 1), (3, 1, 1, 1), torch.int8)
        buf42 = buf40; del buf40  # reuse
        buf112 = empty_strided_cuda((4, 3, 1, 1), (3, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.adaptive_max_pool2d, aten.sigmoid, aten.sigmoid_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_adaptive_max_pool2d_sigmoid_sigmoid_backward_10.run(buf42, buf39, buf41, buf112, 12, grid=grid(12), stream=stream0)
        # Topologically Sorted Source Nodes: [outputs0], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf13, primals_64, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf44 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs0_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf43, primals_65, buf5, buf44, 524288, grid=grid(524288), stream=stream0)
        del primals_65
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [input_35, input_36], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf46, primals_67, 262144, grid=grid(262144), stream=stream0)
        del primals_67
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_37, input_38], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf48, primals_69, 262144, grid=grid(262144), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [outputs0_2], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf21, primals_70, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf50 = reinterpret_tensor(buf43, (4, 64, 32, 32), (65536, 1024, 32, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [outputs0_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_13.run(buf49, primals_71, buf13, buf50, 262144, grid=grid(262144), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [input_39, input_40], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf52, primals_73, 131072, grid=grid(131072), stream=stream0)
        del primals_73
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf54 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [input_41, input_42], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf54, primals_75, 131072, grid=grid(131072), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [outputs0_4], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf29, primals_76, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf56 = reinterpret_tensor(buf49, (4, 128, 16, 16), (32768, 256, 16, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [outputs0_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf55, primals_77, buf21, buf56, 131072, grid=grid(131072), stream=stream0)
        del primals_77
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf58 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_16.run(buf58, primals_79, 65536, grid=grid(65536), stream=stream0)
        del primals_79
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_16.run(buf60, primals_81, 65536, grid=grid(65536), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [outputs0_6], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf37, primals_82, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf62 = reinterpret_tensor(buf55, (4, 256, 8, 8), (16384, 64, 8, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [outputs0_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_17.run(buf61, primals_83, buf29, buf62, 65536, grid=grid(65536), stream=stream0)
        del buf61
        del primals_83
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_47, input_48], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_18.run(buf64, primals_85, 32768, grid=grid(32768), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf66 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [input_49, input_50], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_18.run(buf66, primals_87, 32768, grid=grid(32768), stream=stream0)
        del primals_87
        # Topologically Sorted Source Nodes: [outputs0_8], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf54, primals_88, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf68 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs0_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf67, primals_89, buf5, buf48, buf68, 786432, grid=grid(786432), stream=stream0)
        del buf67
        del primals_89
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf70, primals_91, 262144, grid=grid(262144), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [input_53, input_54], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf72, primals_93, 262144, grid=grid(262144), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [outputs0_11], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf60, primals_94, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf74 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs0_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf73, primals_95, buf13, buf54, buf74, 393216, grid=grid(393216), stream=stream0)
        del buf73
        del primals_95
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [input_55, input_56], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf76, primals_97, 131072, grid=grid(131072), stream=stream0)
        del primals_97
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf78, primals_99, 131072, grid=grid(131072), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [outputs0_14], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf66, primals_100, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf80 = empty_strided_cuda((4, 192, 16, 16), (49152, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs0_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf79, primals_101, buf21, buf60, buf80, 196608, grid=grid(196608), stream=stream0)
        del buf79
        del primals_101
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf82 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [input_59, input_60], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_16.run(buf82, primals_103, 65536, grid=grid(65536), stream=stream0)
        del primals_103
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [input_61, input_62], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_16.run(buf84, primals_105, 65536, grid=grid(65536), stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [outputs0_17], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf78, primals_106, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 16, 64, 64), (65536, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [outputs0_21], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf84, primals_112, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf92 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs0_24], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_22.run(buf91, primals_113, buf13, buf54, buf78, buf92, 524288, grid=grid(524288), stream=stream0)
        del buf91
        del primals_113
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [input_67, input_68], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf94, primals_115, 131072, grid=grid(131072), stream=stream0)
        del primals_115
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [input_69, input_70], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf96, primals_117, 131072, grid=grid(131072), stream=stream0)
        del primals_117
        # Topologically Sorted Source Nodes: [outputs0_25], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_118, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf86 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        buf100 = empty_strided_cuda((4, 80, 64, 64), (327680, 4096, 64, 1), torch.float32)
        buf98 = reinterpret_tensor(buf100, (4, 64, 64, 64), (327680, 4096, 64, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [outputs0_20, outputs0_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf85, primals_107, buf5, buf48, buf72, buf97, primals_119, buf86, buf98, 1048576, grid=grid(1048576), stream=stream0)
        del buf85
        del buf97
        del primals_107
        del primals_119
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf88 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [input_63, input_64], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf88, primals_109, 262144, grid=grid(262144), stream=stream0)
        del primals_109
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf90 = buf89; del buf89  # reuse
        buf99 = reinterpret_tensor(buf100, (4, 16, 64, 64), (327680, 4096, 64, 1), 262144)  # alias
        # Topologically Sorted Source Nodes: [input_65, input_66, outputs0_29], Original ATen: [aten.convolution, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_relu_24.run(buf90, primals_111, buf99, 262144, grid=grid(262144), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_120, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [input_71, input_72], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf102, primals_121, 262144, grid=grid(262144), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [input_73, input_74], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf104, primals_123, 262144, grid=grid(262144), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [final_1], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf48, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [final_2], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf72, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [final_3], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf90, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [final_4], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf104, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf109 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [final_1, final_2, final_3, final_4, add, add_1, add_2], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_25.run(buf109, primals_125, buf106, primals_127, buf107, primals_129, buf108, primals_131, 65536, grid=grid(65536), stream=stream0)
        del buf106
        del buf107
        del primals_125
        del primals_127
        del primals_129
        del primals_131
        buf110 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf109, buf110, 65536, grid=grid(65536), stream=stream0)
        buf111 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_27.run(buf110, buf111, 65536, grid=grid(65536), stream=stream0)
        del buf110
    return (buf111, reinterpret_tensor(buf42, (4, 3), (3, 1), 0), primals_1, primals_2, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_28, primals_29, primals_30, primals_32, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_44, primals_46, primals_47, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_58, primals_59, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, buf1, buf2, buf4, buf5, buf6, buf7, buf9, buf10, buf12, buf13, buf14, buf15, buf17, buf18, buf20, buf21, buf22, buf23, buf25, buf26, buf28, buf29, buf30, buf31, buf33, buf34, buf36, buf37, buf39, buf41, buf44, buf46, buf48, buf50, buf52, buf54, buf56, buf58, buf60, buf62, buf64, buf66, buf68, buf70, buf72, buf74, buf76, buf78, buf80, buf82, buf84, buf86, buf88, buf90, buf92, buf94, buf96, buf100, buf102, buf104, buf111, buf112, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((3, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, 16, 4, 4), (256, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, 32, 4, 4), (512, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((32, 16, 4, 4), (256, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((16, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, 32, 4, 4), (512, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, 16, 4, 4), (256, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((16, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 32, 4, 4), (512, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, 16, 4, 4), (256, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((16, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
