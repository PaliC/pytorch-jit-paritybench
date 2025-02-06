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


# kernel path: inductor_cache/bd/cbdsl5e3p4chuwrbrv4apps77ckjfxwjner5wkrd7ubbyuedtyrc.py
# Topologically Sorted Source Nodes: [out, out1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out => add_1, mul_1, mul_2, sub
#   out1 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %unsqueeze_1), kwargs = {})
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
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 196) % 4)
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


# kernel path: inductor_cache/yx/cyxlflt6nygawyz6vvmp6hzsvz5s3cxixmsrw2esygbpml4fjrzp.py
# Topologically Sorted Source Nodes: [out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_2 => add_3, mul_4, mul_5, sub_1
#   out_3 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bp/cbpntfz62e7lj6chvoa5ufulsy5logew5e473f7t6ltwus364wrf.py
# Topologically Sorted Source Nodes: [out_8, out_9, out1_1], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out1_1 => relu_3
#   out_8 => add_6
#   out_9 => add_8, mul_10, mul_11, sub_3
# Graph fragment:
#   %add_6 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %primals_1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = ((xindex // 196) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr0 + (x0), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wl/cwlemb7jznawu7koxpt4qiyzq7cp7nlu7cakbsiufhr5tsvgnels.py
# Topologically Sorted Source Nodes: [out_17, out_18, out1_2], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   out1_2 => relu_6
#   out_17 => add_13
#   out_18 => add_15, mul_19, mul_20, sub_6
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add_6), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %unsqueeze_318), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 196) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp19, xmask)
    tl.store(out_ptr1 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zx/czxn3whxgs7wkjxdkmhthumnxp5pyrezyl5nit4mkx2w3yy6qubv.py
# Topologically Sorted Source Nodes: [out_mpool1, out_27, out1_3], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   out1_3 => relu_9
#   out_27 => add_22, mul_28, mul_29, sub_9
#   out_mpool1 => _low_memory_max_pool2d_with_offsets, getitem_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_6, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_22,), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_282), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 7) % 7)
    x0 = (xindex % 7)
    x6 = xindex // 7
    x7 = xindex
    x4 = ((xindex // 49) % 4)
    tmp77 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-15) + 2*x0 + 28*x6), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-14) + 2*x0 + 28*x6), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-13) + 2*x0 + 28*x6), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 28*x6), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 28*x6), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 28*x6), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (13 + 2*x0 + 28*x6), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (14 + 2*x0 + 28*x6), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (15 + 2*x0 + 28*x6), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf"))
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
    tmp80 = 1e-05
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
    tl.store(out_ptr0 + (x7), tmp51, xmask)
    tl.store(out_ptr1 + (x7), tmp76, xmask)
    tl.store(out_ptr2 + (x7), tmp93, xmask)
    tl.store(out_ptr3 + (x7), tmp78, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tb/ctbtrxg2os2bzidmpmn7gey6gxygwewzfcnwkqonzlqk2hwl4ml2.py
# Topologically Sorted Source Nodes: [out_29, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_29 => add_24, mul_31, mul_32, sub_10
#   out_30 => relu_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_24,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mw/cmwp4ljibj7noynxhqfsdrtwcpjj3mgrp2hzths2dwytkgblwykg.py
# Topologically Sorted Source Nodes: [out_35, out_36, out1_4], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   out1_4 => relu_12
#   out_35 => add_27
#   out_36 => add_29, mul_37, mul_38, sub_12
# Graph fragment:
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_11, %getitem), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_27, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_27, %unsqueeze_246), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 49) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp19, xmask)
    tl.store(out_ptr1 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vo/cvo3xtokt2z4cdoo7unczknujigzp6lnhq6o3ugjd43jl4xbtbs4.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   interpolate => convert_element_type_31
# Graph fragment:
#   %convert_element_type_31 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.46153846153846156
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/he/chekfx4sdiltelisarzbludc2lm2kzc3lkcjhrslcyiq3e7huu6w.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   interpolate => add_35, clamp_max
# Graph fragment:
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_31, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_35, 6), kwargs = {})
triton_poi_fused_add_clamp_8 = async_compile.triton('triton_poi_fused_add_clamp_8', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.46153846153846156
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 6, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jj/cjjwcacoiewk2k5dzwsq5rrodp2ps7iz36zqlvueagbfhrwmett5.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   interpolate => clamp_max_2, clamp_min, clamp_min_2, convert_element_type_30, iota, mul_45, sub_15
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (14,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, 0.46153846153846156), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_45, 0.0), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_33), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_15, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_9 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_9', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.46153846153846156
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nv/cnveygaltxg3jfaamtv5iml7cngqn5rkxdfgz7plpqvfaeyak7dc.py
# Topologically Sorted Source Nodes: [out_17, out_26, out_35, out_44, interpolate, out_interp1, input_1, input_2], Original ATen: [aten.add, aten._unsafe_index, aten.sub, aten.mul, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_1 => add_42, mul_51, mul_52, sub_20
#   input_2 => relu_15
#   interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_37, add_38, add_39, mul_47, mul_48, mul_49, sub_16, sub_17, sub_19
#   out_17 => add_13
#   out_26 => add_20
#   out_35 => add_27
#   out_44 => add_34
#   out_interp1 => add_40
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add_6), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %add_13), kwargs = {})
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_11, %getitem), kwargs = {})
#   %add_34 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %add_27), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_34, [None, None, %convert_element_type_31, %convert_element_type_33]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_34, [None, None, %convert_element_type_31, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_34, [None, None, %clamp_max, %convert_element_type_33]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_34, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %clamp_max_2), kwargs = {})
#   %add_37 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_47), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %clamp_max_2), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_48), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_38, %add_37), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %clamp_max_3), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %mul_49), kwargs = {})
#   %add_40 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %add_20), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_40, %unsqueeze_121), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_123), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %unsqueeze_125), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_52, %unsqueeze_127), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_42,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_40, %unsqueeze_210), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_native_batch_norm_backward_relu_sub_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_native_batch_norm_backward_relu_sub_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_native_batch_norm_backward_relu_sub_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_native_batch_norm_backward_relu_sub_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = ((xindex // 14) % 14)
    x1 = (xindex % 14)
    x3 = xindex // 196
    x5 = ((xindex // 196) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr11 + (x5), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr12 + (x5), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr13 + (x5), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr14 + (x5), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tl.full([XBLOCK], 7, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tmp11 = tmp10 + tmp6
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr4 + (tmp13 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (tmp13 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (tmp13 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp20 = tmp19 + tmp6
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr4 + (tmp13 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (tmp13 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (tmp13 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp26 = tmp24 + tmp25
    tmp27 = tmp23 + tmp26
    tmp29 = tmp28 + tmp6
    tmp30 = tmp28 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp28)
    tmp32 = tl.load(in_ptr4 + (tmp31 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (tmp31 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (tmp31 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp35 = tmp33 + tmp34
    tmp36 = tmp32 + tmp35
    tmp37 = tmp36 - tmp27
    tmp39 = tmp37 * tmp38
    tmp40 = tmp27 + tmp39
    tmp41 = tl.load(in_ptr4 + (tmp31 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (tmp31 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (tmp31 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp44 = tmp42 + tmp43
    tmp45 = tmp41 + tmp44
    tmp46 = tmp45 - tmp18
    tmp47 = tmp46 * tmp38
    tmp48 = tmp18 + tmp47
    tmp49 = tmp48 - tmp40
    tmp51 = tmp49 * tmp50
    tmp52 = tmp40 + tmp51
    tmp53 = tmp52 + tmp4
    tmp55 = tmp53 - tmp54
    tmp57 = 1e-05
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp66 = tmp64 * tmp65
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr0 + (x0), tmp70, xmask)
    tl.store(out_ptr1 + (x0), tmp55, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qr/cqrqhcwvh2t26y4o4joiyhkxppv5vt3g42fi7zujvzmxcrch6j7f.py
# Topologically Sorted Source Nodes: [input_7, add_1, out_45, out_46, out1_5], Original ATen: [aten.sigmoid, aten.add, aten.mul, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_1 => add_45
#   input_7 => sigmoid
#   out1_5 => relu_17
#   out_45 => mul_56
#   out_46 => add_47, mul_58, mul_59, sub_22
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_16,), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, 1), kwargs = {})
#   %mul_56 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, %add_20), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_56, %unsqueeze_137), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_139), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_141), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_143), kwargs = {})
#   %relu_17 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_56, %unsqueeze_186), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_native_batch_norm_backward_relu_sigmoid_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_native_batch_norm_backward_relu_sigmoid_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_native_batch_norm_backward_relu_sigmoid_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_native_batch_norm_backward_relu_sigmoid_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 196) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = tmp13 * tmp2
    tmp15 = tmp7 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x3), tmp21, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s4/cs4ny6fwc3oe4u4jke6fifcuim2w2kgtby2nj4fhoztyky7ulvln.py
# Topologically Sorted Source Nodes: [input_7, add_1, out_45, out_54], Original ATen: [aten.sigmoid, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_1 => add_45
#   input_7 => sigmoid
#   out_45 => mul_56
#   out_54 => add_52
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_16,), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, 1), kwargs = {})
#   %mul_56 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, %add_20), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_19, %mul_56), kwargs = {})
triton_poi_fused_add_mul_sigmoid_12 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_12(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 14, 14), (784, 196, 14, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_7, (1, ), (1, ))
    assert_size_stride(primals_8, (1, ), (1, ))
    assert_size_stride(primals_9, (1, ), (1, ))
    assert_size_stride(primals_10, (1, ), (1, ))
    assert_size_stride(primals_11, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_12, (1, ), (1, ))
    assert_size_stride(primals_13, (1, ), (1, ))
    assert_size_stride(primals_14, (1, ), (1, ))
    assert_size_stride(primals_15, (1, ), (1, ))
    assert_size_stride(primals_16, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_22, (1, ), (1, ))
    assert_size_stride(primals_23, (1, ), (1, ))
    assert_size_stride(primals_24, (1, ), (1, ))
    assert_size_stride(primals_25, (1, ), (1, ))
    assert_size_stride(primals_26, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_27, (1, ), (1, ))
    assert_size_stride(primals_28, (1, ), (1, ))
    assert_size_stride(primals_29, (1, ), (1, ))
    assert_size_stride(primals_30, (1, ), (1, ))
    assert_size_stride(primals_31, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_32, (4, ), (1, ))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_37, (1, ), (1, ))
    assert_size_stride(primals_38, (1, ), (1, ))
    assert_size_stride(primals_39, (1, ), (1, ))
    assert_size_stride(primals_40, (1, ), (1, ))
    assert_size_stride(primals_41, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_42, (1, ), (1, ))
    assert_size_stride(primals_43, (1, ), (1, ))
    assert_size_stride(primals_44, (1, ), (1, ))
    assert_size_stride(primals_45, (1, ), (1, ))
    assert_size_stride(primals_46, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_47, (4, ), (1, ))
    assert_size_stride(primals_48, (4, ), (1, ))
    assert_size_stride(primals_49, (4, ), (1, ))
    assert_size_stride(primals_50, (4, ), (1, ))
    assert_size_stride(primals_51, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_52, (1, ), (1, ))
    assert_size_stride(primals_53, (1, ), (1, ))
    assert_size_stride(primals_54, (1, ), (1, ))
    assert_size_stride(primals_55, (1, ), (1, ))
    assert_size_stride(primals_56, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_57, (1, ), (1, ))
    assert_size_stride(primals_58, (1, ), (1, ))
    assert_size_stride(primals_59, (1, ), (1, ))
    assert_size_stride(primals_60, (1, ), (1, ))
    assert_size_stride(primals_61, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_62, (4, ), (1, ))
    assert_size_stride(primals_63, (4, ), (1, ))
    assert_size_stride(primals_64, (4, ), (1, ))
    assert_size_stride(primals_65, (4, ), (1, ))
    assert_size_stride(primals_66, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_67, (1, ), (1, ))
    assert_size_stride(primals_68, (1, ), (1, ))
    assert_size_stride(primals_69, (1, ), (1, ))
    assert_size_stride(primals_70, (1, ), (1, ))
    assert_size_stride(primals_71, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_72, (1, ), (1, ))
    assert_size_stride(primals_73, (1, ), (1, ))
    assert_size_stride(primals_74, (1, ), (1, ))
    assert_size_stride(primals_75, (1, ), (1, ))
    assert_size_stride(primals_76, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_77, (4, ), (1, ))
    assert_size_stride(primals_78, (4, ), (1, ))
    assert_size_stride(primals_79, (4, ), (1, ))
    assert_size_stride(primals_80, (4, ), (1, ))
    assert_size_stride(primals_81, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_82, (4, ), (1, ))
    assert_size_stride(primals_83, (4, ), (1, ))
    assert_size_stride(primals_84, (4, ), (1, ))
    assert_size_stride(primals_85, (4, ), (1, ))
    assert_size_stride(primals_86, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_87, (4, ), (1, ))
    assert_size_stride(primals_88, (4, ), (1, ))
    assert_size_stride(primals_89, (4, ), (1, ))
    assert_size_stride(primals_90, (4, ), (1, ))
    assert_size_stride(primals_91, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_92, (1, ), (1, ))
    assert_size_stride(primals_93, (1, ), (1, ))
    assert_size_stride(primals_94, (1, ), (1, ))
    assert_size_stride(primals_95, (1, ), (1, ))
    assert_size_stride(primals_96, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_97, (1, ), (1, ))
    assert_size_stride(primals_98, (1, ), (1, ))
    assert_size_stride(primals_99, (1, ), (1, ))
    assert_size_stride(primals_100, (1, ), (1, ))
    assert_size_stride(primals_101, (4, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 14, 14), (784, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out, out1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(primals_1, primals_2, primals_3, primals_4, primals_5, buf0, 3136, grid=grid(3136), stream=stream0)
        del primals_4
        del primals_5
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 1, 14, 14), (196, 196, 14, 1))
        buf2 = empty_strided_cuda((4, 1, 14, 14), (196, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf1, primals_7, primals_8, primals_9, primals_10, buf2, 784, grid=grid(784), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 1, 14, 14), (196, 196, 14, 1))
        buf4 = empty_strided_cuda((4, 1, 14, 14), (196, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf3, primals_12, primals_13, primals_14, primals_15, buf4, 784, grid=grid(784), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 4, 14, 14), (784, 196, 14, 1))
        buf6 = buf5; del buf5  # reuse
        buf7 = empty_strided_cuda((4, 4, 14, 14), (784, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_8, out_9, out1_1], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2.run(buf6, primals_1, primals_17, primals_18, primals_19, primals_20, buf7, 3136, grid=grid(3136), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_21, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 1, 14, 14), (196, 196, 14, 1))
        buf9 = empty_strided_cuda((4, 1, 14, 14), (196, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf8, primals_22, primals_23, primals_24, primals_25, buf9, 784, grid=grid(784), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 1, 14, 14), (196, 196, 14, 1))
        buf11 = empty_strided_cuda((4, 1, 14, 14), (196, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf10, primals_27, primals_28, primals_29, primals_30, buf11, 784, grid=grid(784), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 4, 14, 14), (784, 196, 14, 1))
        buf13 = empty_strided_cuda((4, 4, 14, 14), (784, 196, 14, 1), torch.float32)
        buf59 = empty_strided_cuda((4, 4, 14, 14), (784, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_17, out_18, out1_2], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_3.run(buf12, buf6, primals_32, primals_33, primals_34, primals_35, buf13, buf59, 3136, grid=grid(3136), stream=stream0)
        del primals_32
        del primals_35
        # Topologically Sorted Source Nodes: [out_19], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 1, 14, 14), (196, 196, 14, 1))
        buf15 = empty_strided_cuda((4, 1, 14, 14), (196, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_20, out_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf14, primals_37, primals_38, primals_39, primals_40, buf15, 784, grid=grid(784), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 1, 14, 14), (196, 196, 14, 1))
        buf17 = empty_strided_cuda((4, 1, 14, 14), (196, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_23, out_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf16, primals_42, primals_43, primals_44, primals_45, buf17, 784, grid=grid(784), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [out_25], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 4, 14, 14), (784, 196, 14, 1))
        buf20 = empty_strided_cuda((4, 4, 7, 7), (196, 49, 7, 1), torch.float32)
        buf21 = empty_strided_cuda((4, 4, 7, 7), (196, 49, 7, 1), torch.int8)
        buf22 = empty_strided_cuda((4, 4, 7, 7), (196, 49, 7, 1), torch.float32)
        buf58 = empty_strided_cuda((4, 4, 7, 7), (196, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_mpool1, out_27, out1_3], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_4.run(buf6, primals_47, primals_48, primals_49, primals_50, buf20, buf21, buf22, buf58, 784, grid=grid(784), stream=stream0)
        del primals_47
        del primals_50
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 1, 7, 7), (49, 49, 7, 1))
        buf24 = empty_strided_cuda((4, 1, 7, 7), (49, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_29, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf23, primals_52, primals_53, primals_54, primals_55, buf24, 196, grid=grid(196), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 1, 7, 7), (49, 49, 7, 1))
        buf26 = empty_strided_cuda((4, 1, 7, 7), (49, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_32, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf25, primals_57, primals_58, primals_59, primals_60, buf26, 196, grid=grid(196), stream=stream0)
        del primals_60
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 4, 7, 7), (196, 49, 7, 1))
        buf28 = empty_strided_cuda((4, 4, 7, 7), (196, 49, 7, 1), torch.float32)
        buf57 = empty_strided_cuda((4, 4, 7, 7), (196, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_35, out_36, out1_4], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_6.run(buf27, buf20, primals_62, primals_63, primals_64, primals_65, buf28, buf57, 784, grid=grid(784), stream=stream0)
        del primals_62
        del primals_65
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 1, 7, 7), (49, 49, 7, 1))
        buf30 = empty_strided_cuda((4, 1, 7, 7), (49, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_38, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf29, primals_67, primals_68, primals_69, primals_70, buf30, 196, grid=grid(196), stream=stream0)
        del primals_70
        # Topologically Sorted Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 1, 7, 7), (49, 49, 7, 1))
        buf32 = empty_strided_cuda((4, 1, 7, 7), (49, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf31, primals_72, primals_73, primals_74, primals_75, buf32, 196, grid=grid(196), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 4, 7, 7), (196, 49, 7, 1))
        buf34 = empty_strided_cuda((14, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf34, 14, grid=grid(14), stream=stream0)
        buf35 = empty_strided_cuda((14, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_8.run(buf35, 14, grid=grid(14), stream=stream0)
        buf36 = empty_strided_cuda((14, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf36, 14, grid=grid(14), stream=stream0)
        buf37 = empty_strided_cuda((14, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_8.run(buf37, 14, grid=grid(14), stream=stream0)
        buf40 = empty_strided_cuda((14, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_9.run(buf40, 14, grid=grid(14), stream=stream0)
        buf42 = empty_strided_cuda((14, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_9.run(buf42, 14, grid=grid(14), stream=stream0)
        buf19 = buf18; del buf18  # reuse
        buf44 = empty_strided_cuda((4, 4, 14, 14), (784, 196, 14, 1), torch.float32)
        buf56 = empty_strided_cuda((4, 4, 14, 14), (784, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_17, out_26, out_35, out_44, interpolate, out_interp1, input_1, input_2], Original ATen: [aten.add, aten._unsafe_index, aten.sub, aten.mul, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_native_batch_norm_backward_relu_sub_10.run(buf19, buf12, buf6, buf35, buf36, buf33, buf27, buf20, buf34, buf37, buf40, buf42, primals_77, primals_78, primals_79, primals_80, buf44, buf56, 3136, grid=grid(3136), stream=stream0)
        del buf20
        del primals_77
        del primals_80
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 4, 14, 14), (784, 196, 14, 1))
        buf46 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf45, primals_82, primals_83, primals_84, primals_85, buf46, 3136, grid=grid(3136), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 4, 14, 14), (784, 196, 14, 1))
        buf48 = empty_strided_cuda((4, 4, 14, 14), (784, 196, 14, 1), torch.float32)
        buf55 = empty_strided_cuda((4, 4, 14, 14), (784, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, add_1, out_45, out_46, out1_5], Original ATen: [aten.sigmoid, aten.add, aten.mul, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_native_batch_norm_backward_relu_sigmoid_11.run(buf47, buf19, primals_87, primals_88, primals_89, primals_90, buf48, buf55, 3136, grid=grid(3136), stream=stream0)
        del primals_87
        del primals_90
        # Topologically Sorted Source Nodes: [out_47], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 1, 14, 14), (196, 196, 14, 1))
        buf50 = reinterpret_tensor(buf33, (4, 1, 14, 14), (196, 196, 14, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [out_48, out_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf49, primals_92, primals_93, primals_94, primals_95, buf50, 784, grid=grid(784), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 1, 14, 14), (196, 196, 14, 1))
        buf52 = reinterpret_tensor(buf27, (4, 1, 14, 14), (196, 196, 14, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [out_51, out_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf51, primals_97, primals_98, primals_99, primals_100, buf52, 784, grid=grid(784), stream=stream0)
        del primals_100
        # Topologically Sorted Source Nodes: [out_53], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 4, 14, 14), (784, 196, 14, 1))
        buf54 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [input_7, add_1, out_45, out_54], Original ATen: [aten.sigmoid, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_12.run(buf54, buf47, buf19, 3136, grid=grid(3136), stream=stream0)
    return (buf54, primals_1, primals_2, primals_3, primals_6, primals_7, primals_8, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_29, primals_31, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_46, primals_48, primals_49, primals_51, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_59, primals_61, primals_63, primals_64, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_76, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_99, primals_101, buf0, buf1, buf2, buf3, buf4, buf6, buf7, buf8, buf9, buf10, buf11, buf13, buf14, buf15, buf16, buf17, buf19, buf21, buf22, buf23, buf24, buf25, buf26, buf28, buf29, buf30, buf31, buf32, buf34, buf35, buf36, buf37, buf40, buf42, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf55, buf56, buf57, buf58, buf59, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 14, 14), (784, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
