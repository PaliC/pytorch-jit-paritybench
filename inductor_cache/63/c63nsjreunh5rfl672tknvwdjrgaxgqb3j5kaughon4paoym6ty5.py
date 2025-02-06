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


# kernel path: inductor_cache/43/c43nk47j4oy5ovrledtpzrgolsb2twwwd6fwmk5n5p7ypru7swez.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_1, mul_1, mul_2, sub
#   input_2 => relu
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 4)
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


# kernel path: inductor_cache/q2/cq2ioooxqiexcae5net4by4ywxxwy5awpgbrtd3emcxia7y6eyai.py
# Topologically Sorted Source Nodes: [input_3, input_4, input_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_3 => convolution
#   input_4 => add_3, mul_4, mul_5, sub_1
#   input_5 => relu_1
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_6, %primals_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 2)
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


# kernel path: inductor_cache/y7/cy7a6za5kamdvmv45t3d3qeqzx2udu2sje3ggobvdifg4xed6zk7.py
# Topologically Sorted Source Nodes: [input_9, up1, input_10, input_11], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_8, mul_10, mul_11, sub_3
#   input_11 => relu_3
#   input_9 => convolution_2
#   up1 => add_6
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_18, %primals_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %primals_1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/6l/c6lgfobamw6fqmpp7txmpiwmnad45d2vkmsvadwrapk77vwbs2g2.py
# Topologically Sorted Source Nodes: [up1, input_18, up1_1, input_19, input_20], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_18 => convolution_5
#   input_19 => add_15, mul_19, mul_20, sub_6
#   input_20 => relu_6
#   up1 => add_6
#   up1_1 => add_13
# Graph fragment:
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %primals_1), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_36, %primals_37, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add_6), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp4 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/yy/cyyvuycre34pkdt5gqfhsyon5ntffk6grdmce45h2podeb3lrza5.py
# Topologically Sorted Source Nodes: [up1, input_18, up1_1, input_27, up1_2, input_28, input_29], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_18 => convolution_5
#   input_27 => convolution_8
#   input_28 => add_22, mul_28, mul_29, sub_9
#   input_29 => relu_9
#   up1 => add_6
#   up1_1 => add_13
#   up1_2 => add_20
# Graph fragment:
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %primals_1), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_36, %primals_37, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add_6), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_54, %primals_55, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %add_13), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_20, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_22,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x3), None)
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(in_out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/ye/cye4da7utlqsnlst4fxs6nyyr62gmwzyowwwberjzds6rbadyjjp.py
# Topologically Sorted Source Nodes: [low1, input_37, input_38], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_37 => add_29, mul_37, mul_38, sub_12
#   input_38 => relu_12
#   low1 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%primals_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x4 = xindex // 32
    x2 = ((xindex // 1024) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x5), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/37/c37e65le7mxhscendl2w5cwe4bnbauaecuci4mn6ek3wntzi74qi.py
# Topologically Sorted Source Nodes: [input_39, input_40, input_41], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_39 => convolution_12
#   input_40 => add_31, mul_40, mul_41, sub_13
#   input_41 => relu_13
# Graph fragment:
#   %convolution_12 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %primals_78, %primals_79, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 2)
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


# kernel path: inductor_cache/6b/c6bi56ira7wmpzsww7rh5za6gt6m7fwwegqx6zufn6rh64mv7laa.py
# Topologically Sorted Source Nodes: [low1, input_45, low1_1, input_46, input_47], Original ATen: [aten.max_pool2d_with_indices, aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_45 => convolution_14
#   input_46 => add_36, mul_46, mul_47, sub_15
#   input_47 => relu_15
#   low1 => _low_memory_max_pool2d_with_offsets
#   low1_1 => add_34
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%primals_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %primals_90, %primals_91, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %getitem), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_36,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x2 = ((xindex // 1024) % 4)
    x0 = (xindex % 32)
    x6 = xindex // 32
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (2*x0 + 128*x6), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 2*x0 + 128*x6), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (64 + 2*x0 + 128*x6), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (65 + 2*x0 + 128*x6), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(in_out_ptr0 + (x5), tmp10, None)
    tl.store(out_ptr0 + (x5), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/fa/cfag5ky6ntfazr4ecpxce6jlstkl7ygene4snxpnzol4dtfrxrpz.py
# Topologically Sorted Source Nodes: [input_54, low1_2, input_55, input_56], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_54 => convolution_17
#   input_55 => add_43, mul_55, mul_56, sub_18
#   input_56 => relu_18
#   low1_2 => add_41
# Graph fragment:
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_108, %primals_109, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_41 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_34), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_41, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %relu_18 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_43,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/7b/c7bcgywfzfdpjr7oi4ud7p3qnjlplgvmgrgq7etsdvqjfqnuniip.py
# Topologically Sorted Source Nodes: [input_54, low1_2, input_63, low1_3, input_64, input_65], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_54 => convolution_17
#   input_63 => convolution_20
#   input_64 => add_50, mul_64, mul_65, sub_21
#   input_65 => relu_21
#   low1_2 => add_41
#   low1_3 => add_48
# Graph fragment:
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_108, %primals_109, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_41 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_34), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %primals_126, %primals_127, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_20, %add_41), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_48, %unsqueeze_169), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_173), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_175), kwargs = {})
#   %relu_21 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_50,), kwargs = {})
#   %sub_290 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_48, %unsqueeze_2862), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp8, None)
    tl.store(out_ptr0 + (x3), tmp25, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/6o/c6occny2yvxc6erd4ku2a6obob6skjlm7de6phit74pdtsq3zhdn.py
# Topologically Sorted Source Nodes: [input_72, low1_4, input_73, input_74], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_72 => convolution_23
#   input_73 => add_57, mul_73, mul_74, sub_24
#   input_74 => relu_24
#   low1_4 => add_55
# Graph fragment:
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %primals_144, %primals_145, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_55 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_23, %add_48), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_55, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_57,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/mj/cmjga4skacx7vw6pixdkf5petaqv2fr6mqppwpgzqbqxs2braa2x.py
# Topologically Sorted Source Nodes: [input_81, up1_4, input_90, up1_5, input_91, input_92], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_81 => convolution_26
#   input_90 => convolution_29
#   input_91 => add_71, mul_91, mul_92, sub_30
#   input_92 => relu_30
#   up1_4 => add_62
#   up1_5 => add_69
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_26, %primals_162, %primals_163, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_62 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %add_55), kwargs = {})
#   %convolution_29 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_29, %primals_180, %primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_69 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_29, %add_62), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_69, %unsqueeze_241), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %relu_30 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_71,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp8, None)
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/xa/cxatxxydcnleescp3x4d2jqha5ddtd4h3ptslmwuyf7dehhymgpf.py
# Topologically Sorted Source Nodes: [low1_5, input_109, input_110], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_109 => add_85, mul_109, mul_110, sub_36
#   input_110 => relu_36
#   low1_5 => _low_memory_max_pool2d_with_offsets_1, getitem_3
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_55, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_2, %unsqueeze_289), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_293), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_295), kwargs = {})
#   %relu_36 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_85,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x5 = xindex
    x3 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x5), tmp15, None)
    tl.store(out_ptr1 + (x5), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/cy/ccyutz6mdqxks3upalupwuitbfkjpfbuhlcc5wuo7uhwoshq32f4.py
# Topologically Sorted Source Nodes: [input_111, input_112, input_113], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_111 => convolution_36
#   input_112 => add_87, mul_112, mul_113, sub_37
#   input_113 => relu_37
# Graph fragment:
#   %convolution_36 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_36, %primals_222, %primals_223, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_297), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %relu_37 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_87,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 2)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wb/cwb4z45zajz3f5cmihi3ucojxzh5uha3izsjoqqrye2uzcki336j.py
# Topologically Sorted Source Nodes: [low1_5, input_117, low1_6, input_118, input_119], Original ATen: [aten.max_pool2d_with_indices, aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_117 => convolution_38
#   input_118 => add_92, mul_118, mul_119, sub_39
#   input_119 => relu_39
#   low1_5 => _low_memory_max_pool2d_with_offsets_1
#   low1_6 => add_90
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_55, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_38 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_38, %primals_234, %primals_235, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_90 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_38, %getitem_2), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_90, %unsqueeze_313), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %relu_39 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_92,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x2 = ((xindex // 256) % 4)
    x0 = (xindex % 16)
    x6 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (2*x0 + 64*x6), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 2*x0 + 64*x6), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (32 + 2*x0 + 64*x6), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (33 + 2*x0 + 64*x6), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(in_out_ptr0 + (x5), tmp10, None)
    tl.store(out_ptr0 + (x5), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/jx/cjxmvzia7ucy53gcuto5aw2vws6wilyqr4nd5gmdrfydecnmpaed.py
# Topologically Sorted Source Nodes: [input_126, low1_7, input_127, input_128], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_126 => convolution_41
#   input_127 => add_99, mul_127, mul_128, sub_42
#   input_128 => relu_42
#   low1_7 => add_97
# Graph fragment:
#   %convolution_41 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %primals_252, %primals_253, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_97 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_41, %add_90), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_97, %unsqueeze_337), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_341), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_343), kwargs = {})
#   %relu_42 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_99,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/z6/cz66vmjxlbdzxxmb3fdlc2sxbwjehfqkjuxu3npgrnmr5ueg3emq.py
# Topologically Sorted Source Nodes: [input_126, low1_7, input_135, low1_8, input_136, input_137], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_126 => convolution_41
#   input_135 => convolution_44
#   input_136 => add_106, mul_136, mul_137, sub_45
#   input_137 => relu_45
#   low1_7 => add_97
#   low1_8 => add_104
# Graph fragment:
#   %convolution_41 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %primals_252, %primals_253, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_97 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_41, %add_90), kwargs = {})
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_44, %primals_270, %primals_271, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_104 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_44, %add_97), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_104, %unsqueeze_361), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %unsqueeze_365), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_137, %unsqueeze_367), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_106,), kwargs = {})
#   %sub_266 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_104, %unsqueeze_2574), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp8, None)
    tl.store(out_ptr0 + (x3), tmp25, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/ax/caxwzjvnlimfl44ep56pzzlai2vnszg2n4qewgvg2xowpisg7pve.py
# Topologically Sorted Source Nodes: [input_144, low1_9, input_145, input_146], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_144 => convolution_47
#   input_145 => add_113, mul_145, mul_146, sub_48
#   input_146 => relu_48
#   low1_9 => add_111
# Graph fragment:
#   %convolution_47 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_47, %primals_288, %primals_289, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_111 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_47, %add_104), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_111, %unsqueeze_385), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_387), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_145, %unsqueeze_389), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_146, %unsqueeze_391), kwargs = {})
#   %relu_48 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_113,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/sq/csqswano7xgty65y3hskp4rknayzynhmu6yy5ciqsiljaqomuosx.py
# Topologically Sorted Source Nodes: [input_153, up1_8, input_162, up1_9, input_163, input_164], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_153 => convolution_50
#   input_162 => convolution_53
#   input_163 => add_127, mul_163, mul_164, sub_54
#   input_164 => relu_54
#   up1_8 => add_118
#   up1_9 => add_125
# Graph fragment:
#   %convolution_50 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_50, %primals_306, %primals_307, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_118 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_50, %add_111), kwargs = {})
#   %convolution_53 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_53, %primals_324, %primals_325, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_125 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_53, %add_118), kwargs = {})
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_125, %unsqueeze_433), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_435), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %unsqueeze_437), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %unsqueeze_439), kwargs = {})
#   %relu_54 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_127,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp8, None)
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/4d/c4drnipev73hqhg54ksmwkv5vjtn5tdzqyt2qitao5pjplt7twwj.py
# Topologically Sorted Source Nodes: [low1_10, input_181, input_182], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_181 => add_141, mul_181, mul_182, sub_60
#   input_182 => relu_60
#   low1_10 => _low_memory_max_pool2d_with_offsets_2, getitem_5
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_111, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_4, %unsqueeze_481), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_483), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_181, %unsqueeze_485), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %unsqueeze_487), kwargs = {})
#   %relu_60 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_141,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x5 = xindex
    x3 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x5), tmp15, xmask)
    tl.store(out_ptr1 + (x5), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m3/cm3ltm62qevl5yqkklqxfl7gcane5ribknk7h5vflg2zn4v4tpy3.py
# Topologically Sorted Source Nodes: [input_183, input_184, input_185], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_183 => convolution_60
#   input_184 => add_143, mul_184, mul_185, sub_61
#   input_185 => relu_61
# Graph fragment:
#   %convolution_60 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_60, %primals_366, %primals_367, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_60, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_143,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 2)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jt/cjtpqw6z3rugqsdzncfy234b55ywytyhxozcyfneir3kg3ggsvx7.py
# Topologically Sorted Source Nodes: [low1_10, input_189, low1_11, input_190, input_191], Original ATen: [aten.max_pool2d_with_indices, aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_189 => convolution_62
#   input_190 => add_148, mul_190, mul_191, sub_63
#   input_191 => relu_63
#   low1_10 => _low_memory_max_pool2d_with_offsets_2
#   low1_11 => add_146
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_111, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_62 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_62, %primals_378, %primals_379, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_146 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_62, %getitem_4), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_146, %unsqueeze_505), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_509), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_511), kwargs = {})
#   %relu_63 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_148,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex
    x2 = ((xindex // 64) % 4)
    x0 = (xindex % 8)
    x6 = xindex // 8
    tmp0 = tl.load(in_out_ptr0 + (x5), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (2*x0 + 32*x6), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 2*x0 + 32*x6), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (16 + 2*x0 + 32*x6), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (17 + 2*x0 + 32*x6), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(in_out_ptr0 + (x5), tmp10, xmask)
    tl.store(out_ptr0 + (x5), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/iy/ciydyxbslan6d3ndbkf6e4mpriv4zy4op4otedutlt4t355iozms.py
# Topologically Sorted Source Nodes: [input_198, low1_12, input_199, input_200], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_198 => convolution_65
#   input_199 => add_155, mul_199, mul_200, sub_66
#   input_200 => relu_66
#   low1_12 => add_153
# Graph fragment:
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_65, %primals_396, %primals_397, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_153 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_65, %add_146), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_153, %unsqueeze_529), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_533), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_535), kwargs = {})
#   %relu_66 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ui/cuimgircsoywmhv5o43e2z25u4xnwyeo7auntdflehpw7nl44qay.py
# Topologically Sorted Source Nodes: [input_198, low1_12, input_207, low1_13, input_208, input_209], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_198 => convolution_65
#   input_207 => convolution_68
#   input_208 => add_162, mul_208, mul_209, sub_69
#   input_209 => relu_69
#   low1_12 => add_153
#   low1_13 => add_160
# Graph fragment:
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_65, %primals_396, %primals_397, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_153 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_65, %add_146), kwargs = {})
#   %convolution_68 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_68, %primals_414, %primals_415, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_160 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_68, %add_153), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_160, %unsqueeze_553), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_208, %unsqueeze_557), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_209, %unsqueeze_559), kwargs = {})
#   %relu_69 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_162,), kwargs = {})
#   %sub_242 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_160, %unsqueeze_2286), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr0 + (x3), tmp25, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5n/c5nfa5msnjtxxtef72typxnjcqbm77vawhi6gjts6eei6es6ksbf.py
# Topologically Sorted Source Nodes: [input_216, low1_14, input_217, input_218], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_216 => convolution_71
#   input_217 => add_169, mul_217, mul_218, sub_72
#   input_218 => relu_72
#   low1_14 => add_167
# Graph fragment:
#   %convolution_71 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_71, %primals_432, %primals_433, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_167 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_71, %add_160), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_167, %unsqueeze_577), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_581), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_583), kwargs = {})
#   %relu_72 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_169,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k7/ck726bsbiln5kmv7eytktlz6mguimalifrggkfju2zvwfhk3jndr.py
# Topologically Sorted Source Nodes: [input_225, up1_12, input_234, up1_13, input_235, input_236], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_225 => convolution_74
#   input_234 => convolution_77
#   input_235 => add_183, mul_235, mul_236, sub_78
#   input_236 => relu_78
#   up1_12 => add_174
#   up1_13 => add_181
# Graph fragment:
#   %convolution_74 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_74, %primals_450, %primals_451, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_174 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_74, %add_167), kwargs = {})
#   %convolution_77 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_77, %primals_468, %primals_469, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_181 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_77, %add_174), kwargs = {})
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_181, %unsqueeze_625), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %unsqueeze_629), kwargs = {})
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %unsqueeze_631), kwargs = {})
#   %relu_78 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_183,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr0 + (x3), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wu/cwurtva3ahf2ctps7kdiawabx3lpmttpqywlztzptz7bl7vadaag.py
# Topologically Sorted Source Nodes: [low1_15, input_253, input_254], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_253 => add_197, mul_253, mul_254, sub_84
#   input_254 => relu_84
#   low1_15 => _low_memory_max_pool2d_with_offsets_3, getitem_7
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_167, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_6, %unsqueeze_673), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_253, %unsqueeze_677), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_254, %unsqueeze_679), kwargs = {})
#   %relu_84 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_197,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x5 = xindex
    x3 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x5), tmp15, xmask)
    tl.store(out_ptr1 + (x5), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u2/cu27g3huzuxbuszjuum33vcg5et6d3pbaoqhgncycxos4zyczsft.py
# Topologically Sorted Source Nodes: [input_255, input_256, input_257], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_255 => convolution_84
#   input_256 => add_199, mul_256, mul_257, sub_85
#   input_257 => relu_85
# Graph fragment:
#   %convolution_84 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_84, %primals_510, %primals_511, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_681), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_685), kwargs = {})
#   %add_199 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_687), kwargs = {})
#   %relu_85 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_199,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5t/c5tc4b2qljduqxv4wyyzoagzlaqjvtuxl6o5hxtf32wuultijdsi.py
# Topologically Sorted Source Nodes: [low1_15, input_261, low1_16, input_262, input_263], Original ATen: [aten.max_pool2d_with_indices, aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_261 => convolution_86
#   input_262 => add_204, mul_262, mul_263, sub_87
#   input_263 => relu_87
#   low1_15 => _low_memory_max_pool2d_with_offsets_3
#   low1_16 => add_202
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_167, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_86 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_86, %primals_522, %primals_523, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_202 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_86, %getitem_6), kwargs = {})
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_202, %unsqueeze_697), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_701), kwargs = {})
#   %add_204 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_703), kwargs = {})
#   %relu_87 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_204,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex
    x2 = ((xindex // 16) % 4)
    x0 = (xindex % 4)
    x6 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x5), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (2*x0 + 16*x6), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 2*x0 + 16*x6), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (8 + 2*x0 + 16*x6), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (9 + 2*x0 + 16*x6), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(in_out_ptr0 + (x5), tmp10, xmask)
    tl.store(out_ptr0 + (x5), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p7/cp73cpxi2tc23nbm6i2e7lqf7kpw7q73bn64li63ikrbi4n3ufjn.py
# Topologically Sorted Source Nodes: [input_270, low1_17, input_271, input_272], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_270 => convolution_89
#   input_271 => add_211, mul_271, mul_272, sub_90
#   input_272 => relu_90
#   low1_17 => add_209
# Graph fragment:
#   %convolution_89 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_89, %primals_540, %primals_541, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_209 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_89, %add_202), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_209, %unsqueeze_721), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %unsqueeze_725), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %unsqueeze_727), kwargs = {})
#   %relu_90 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_211,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sw/csw777olk3tqc2ptevbljwmkrtuudjetbpq6mo3sdcvjk5oqwiur.py
# Topologically Sorted Source Nodes: [input_270, low1_17, input_279, low1_18, input_280, input_281], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_270 => convolution_89
#   input_279 => convolution_92
#   input_280 => add_218, mul_280, mul_281, sub_93
#   input_281 => relu_93
#   low1_17 => add_209
#   low1_18 => add_216
# Graph fragment:
#   %convolution_89 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_89, %primals_540, %primals_541, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_209 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_89, %add_202), kwargs = {})
#   %convolution_92 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_92, %primals_558, %primals_559, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_216 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_92, %add_209), kwargs = {})
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_216, %unsqueeze_745), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %unsqueeze_749), kwargs = {})
#   %add_218 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_281, %unsqueeze_751), kwargs = {})
#   %relu_93 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_218,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr0 + (x3), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n5/cn54ldojw4pmcdas2qu7kew2mr6dpdnn63qjpxibr2zaqme2e4tk.py
# Topologically Sorted Source Nodes: [input_342, low3_1, input_351, low3_2, input_352, input_353], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_342 => convolution_113
#   input_351 => convolution_116
#   input_352 => add_274, mul_352, mul_353, sub_117
#   input_353 => relu_117
#   low3_1 => add_265
#   low3_2 => add_272
# Graph fragment:
#   %convolution_113 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_113, %primals_684, %primals_685, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_265 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_113, %add_258), kwargs = {})
#   %convolution_116 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_116, %primals_702, %primals_703, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_272 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_116, %add_265), kwargs = {})
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_272, %unsqueeze_937), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_941), kwargs = {})
#   %add_274 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_943), kwargs = {})
#   %relu_117 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_274,), kwargs = {})
#   %sub_194 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_272, %unsqueeze_1710), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr0 + (x3), tmp25, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7g/c7gnyt2atawn4d3s4es6fs4eg6kx7h4mkw7hctywukan3wnoekq3.py
# Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   up2 => add_280, add_281, convert_element_type_240, convert_element_type_241, iota, mul_360, mul_361
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_360 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_280 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_360, 0), kwargs = {})
#   %convert_element_type_240 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_280, torch.float32), kwargs = {})
#   %add_281 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_240, 0.0), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_281, 0.5), kwargs = {})
#   %convert_element_type_241 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_361, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_32 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_32(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/et/cetvkilyysl6yvo3o3hgdsozpkfqrxpv65befiwmyj7n7efjkhzq.py
# Topologically Sorted Source Nodes: [input_243, up1_14, input_252, up1_15, input_360, low3_3, up2, low2_4, input_361, input_362], Original ATen: [aten.convolution, aten.add, aten._unsafe_index, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_243 => convolution_80
#   input_252 => convolution_83
#   input_360 => convolution_119
#   input_361 => add_286, mul_365, mul_366, sub_120
#   input_362 => relu_120
#   low2_4 => add_284
#   low3_3 => add_279
#   up1_14 => add_188
#   up1_15 => add_195
#   up2 => _unsafe_index
# Graph fragment:
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_80, %primals_486, %primals_487, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_188 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_80, %add_181), kwargs = {})
#   %convolution_83 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_83, %primals_504, %primals_505, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_195 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_83, %add_188), kwargs = {})
#   %convolution_119 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_119, %primals_720, %primals_721, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_279 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_119, %add_272), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_279, [None, None, %unsqueeze_960, %convert_element_type_241]), kwargs = {})
#   %add_284 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_195, %_unsafe_index), kwargs = {})
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_284, %unsqueeze_962), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_964), kwargs = {})
#   %mul_366 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_365, %unsqueeze_966), kwargs = {})
#   %add_286 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_366, %unsqueeze_968), kwargs = {})
#   %relu_120 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_286,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex
    x2 = ((xindex // 64) % 4)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x6 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x5), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x5), xmask)
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x5), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tl.full([XBLOCK], 4, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tmp15 = tmp14 + tmp10
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr5 + (tmp17 + 4*tmp13 + 16*x6), xmask, eviction_policy='evict_last')
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr7 + (tmp17 + 4*tmp13 + 16*x6), xmask, eviction_policy='evict_last')
    tmp22 = tmp20 + tmp21
    tmp23 = tmp8 + tmp22
    tmp25 = tmp23 - tmp24
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tl.full([1], 1, tl.int32)
    tmp31 = tmp30 / tmp29
    tmp32 = 1.0
    tmp33 = tmp31 * tmp32
    tmp34 = tmp25 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tl.full([1], 0, tl.int32)
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tl.store(in_out_ptr0 + (x5), tmp23, xmask)
    tl.store(out_ptr0 + (x5), tmp40, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t2/ct2kdmoc63wo6fn6h23bmqny3iib5imnsnoltejsass4gtmmrjsj.py
# Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   up2_1 => add_313, add_314, convert_element_type_268, convert_element_type_269, iota_2, mul_400, mul_401
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_400 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_313 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_400, 0), kwargs = {})
#   %convert_element_type_268 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_313, torch.float32), kwargs = {})
#   %add_314 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_268, 0.0), kwargs = {})
#   %mul_401 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_314, 0.5), kwargs = {})
#   %convert_element_type_269 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_401, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_34 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_34(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/fe/cfe5jihs2eund4hh5twjvrmk2hbmey2sb6tgwpbnxuhxagyufyt2.py
# Topologically Sorted Source Nodes: [input_171, up1_10, input_180, up1_11, input_387, low3_6, input_396, low3_7, up2_1, low2_5, input_397, input_398], Original ATen: [aten.convolution, aten.add, aten._unsafe_index, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_171 => convolution_56
#   input_180 => convolution_59
#   input_387 => convolution_128
#   input_396 => convolution_131
#   input_397 => add_319, mul_405, mul_406, sub_132
#   input_398 => relu_132
#   low2_5 => add_317
#   low3_6 => add_305
#   low3_7 => add_312
#   up1_10 => add_132
#   up1_11 => add_139
#   up2_1 => _unsafe_index_1
# Graph fragment:
#   %convolution_56 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_56, %primals_342, %primals_343, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_132 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_56, %add_125), kwargs = {})
#   %convolution_59 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_59, %primals_360, %primals_361, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_59, %add_132), kwargs = {})
#   %convolution_128 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_128, %primals_774, %primals_775, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_305 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_128, %add_298), kwargs = {})
#   %convolution_131 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_131, %primals_792, %primals_793, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_312 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_131, %add_305), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_312, [None, None, %unsqueeze_1057, %convert_element_type_269]), kwargs = {})
#   %add_317 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_139, %_unsafe_index_1), kwargs = {})
#   %sub_132 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_317, %unsqueeze_1059), kwargs = {})
#   %mul_405 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_132, %unsqueeze_1061), kwargs = {})
#   %mul_406 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_405, %unsqueeze_1063), kwargs = {})
#   %add_319 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_406, %unsqueeze_1065), kwargs = {})
#   %relu_132 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_319,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x2 = ((xindex // 256) % 4)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x5), None)
    tmp4 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x5), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr10 + (x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr11 + (x2), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr12 + (x2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tl.full([XBLOCK], 8, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tmp15 = tmp14 + tmp10
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr5 + (tmp17 + 8*tmp13 + 64*x6), None, eviction_policy='evict_last')
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr7 + (tmp17 + 8*tmp13 + 64*x6), None, eviction_policy='evict_last')
    tmp23 = tmp21 + tmp22
    tmp24 = tl.load(in_ptr9 + (tmp17 + 8*tmp13 + 64*x6), None, eviction_policy='evict_last')
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 + tmp25
    tmp27 = tmp8 + tmp26
    tmp29 = tmp27 - tmp28
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full([1], 0, tl.int32)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tl.store(in_out_ptr0 + (x5), tmp27, None)
    tl.store(out_ptr0 + (x5), tmp44, None)
''', device_str='cuda')


# kernel path: inductor_cache/y7/cy7vrqzl7hsk4qhwf5gqopakrdtsjfzmzpfub5dmd22ffnlon37b.py
# Topologically Sorted Source Nodes: [input_405, low3_8, input_414, low3_9, input_415, input_416], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_405 => convolution_134
#   input_414 => convolution_137
#   input_415 => add_333, mul_423, mul_424, sub_138
#   input_416 => relu_138
#   low3_8 => add_324
#   low3_9 => add_331
# Graph fragment:
#   %convolution_134 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_134, %primals_810, %primals_811, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_324 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_134, %add_317), kwargs = {})
#   %convolution_137 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_137, %primals_828, %primals_829, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_331 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_137, %add_324), kwargs = {})
#   %sub_138 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_331, %unsqueeze_1107), kwargs = {})
#   %mul_423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_138, %unsqueeze_1109), kwargs = {})
#   %mul_424 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_423, %unsqueeze_1111), kwargs = {})
#   %add_333 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_424, %unsqueeze_1113), kwargs = {})
#   %relu_138 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_333,), kwargs = {})
#   %sub_173 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_331, %unsqueeze_1458), kwargs = {})
#   %sub_176 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_324, %unsqueeze_1494), kwargs = {})
#   %sub_179 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_317, %unsqueeze_1530), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp7 - tmp9
    tmp12 = tmp6 - tmp11
    tmp14 = tmp8 - tmp13
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tl.full([1], 1, tl.int32)
    tmp20 = tmp19 / tmp18
    tmp21 = 1.0
    tmp22 = tmp20 * tmp21
    tmp23 = tmp14 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full([1], 0, tl.int32)
    tmp29 = triton_helpers.maximum(tmp28, tmp27)
    tl.store(in_out_ptr0 + (x3), tmp8, None)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp12, None)
    tl.store(out_ptr2 + (x3), tmp29, None)
    tl.store(out_ptr3 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/53/c53ckueqxmijre3rrrbcihtxhobj5373sfkch36n6fvr3e7ajvzo.py
# Topologically Sorted Source Nodes: [input_423, low3_10, input_424, input_425], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_423 => convolution_140
#   input_424 => add_340, mul_432, mul_433, sub_141
#   input_425 => relu_141
#   low3_10 => add_338
# Graph fragment:
#   %convolution_140 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_140, %primals_846, %primals_847, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_338 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_140, %add_331), kwargs = {})
#   %sub_141 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_338, %unsqueeze_1131), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_141, %unsqueeze_1133), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_432, %unsqueeze_1135), kwargs = {})
#   %add_340 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_433, %unsqueeze_1137), kwargs = {})
#   %relu_141 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_340,), kwargs = {})
#   %sub_170 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_338, %unsqueeze_1422), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x3), tmp21, None)
    tl.store(out_ptr1 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/bz/cbz56fmcrin4xrbfuptgv5skpy4iehosijcqgzspluujlmace262.py
# Topologically Sorted Source Nodes: [up2_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   up2_2 => add_346, add_347, convert_element_type_296, convert_element_type_297, iota_4, mul_440, mul_441
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_346 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_440, 0), kwargs = {})
#   %convert_element_type_296 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_346, torch.float32), kwargs = {})
#   %add_347 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_296, 0.0), kwargs = {})
#   %mul_441 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_347, 0.5), kwargs = {})
#   %convert_element_type_297 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_441, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_38 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_38(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/n3/cn3cwelu4vtaoz33gbznqkcdtdb5hogmqvktkecehr5m67p53qy7.py
# Topologically Sorted Source Nodes: [input_99, up1_6, input_108, up1_7, input_423, low3_10, input_432, low3_11, up2_2, low2_6, input_433, input_434], Original ATen: [aten.convolution, aten.add, aten._unsafe_index, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_108 => convolution_35
#   input_423 => convolution_140
#   input_432 => convolution_143
#   input_433 => add_352, mul_445, mul_446, sub_144
#   input_434 => relu_144
#   input_99 => convolution_32
#   low2_6 => add_350
#   low3_10 => add_338
#   low3_11 => add_345
#   up1_6 => add_76
#   up1_7 => add_83
#   up2_2 => _unsafe_index_2
# Graph fragment:
#   %convolution_32 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_32, %primals_198, %primals_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_76 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_32, %add_69), kwargs = {})
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_35, %primals_216, %primals_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_35, %add_76), kwargs = {})
#   %convolution_140 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_140, %primals_846, %primals_847, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_338 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_140, %add_331), kwargs = {})
#   %convolution_143 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_143, %primals_864, %primals_865, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_345 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_143, %add_338), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_345, [None, None, %unsqueeze_1154, %convert_element_type_297]), kwargs = {})
#   %add_350 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_83, %_unsafe_index_2), kwargs = {})
#   %sub_144 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_350, %unsqueeze_1156), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_144, %unsqueeze_1158), kwargs = {})
#   %mul_446 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_445, %unsqueeze_1160), kwargs = {})
#   %add_352 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_446, %unsqueeze_1162), kwargs = {})
#   %relu_144 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_352,), kwargs = {})
#   %sub_167 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_350, %unsqueeze_1386), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_native_batch_norm_backward_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_native_batch_norm_backward_relu_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_native_batch_norm_backward_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_native_batch_norm_backward_relu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x2 = ((xindex // 1024) % 4)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x6 = xindex // 1024
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x5), None)
    tmp4 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x5), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr10 + (x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr11 + (x2), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr12 + (x2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tl.full([XBLOCK], 16, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tmp15 = tmp14 + tmp10
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr5 + (tmp17 + 16*tmp13 + 256*x6), None, eviction_policy='evict_last')
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr7 + (tmp17 + 16*tmp13 + 256*x6), None, eviction_policy='evict_last')
    tmp23 = tmp21 + tmp22
    tmp24 = tl.load(in_ptr9 + (tmp17 + 16*tmp13 + 256*x6), None, eviction_policy='evict_last')
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 + tmp25
    tmp27 = tmp8 + tmp26
    tmp29 = tmp27 - tmp28
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full([1], 0, tl.int32)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tl.store(in_out_ptr0 + (x5), tmp27, None)
    tl.store(out_ptr0 + (x5), tmp44, None)
    tl.store(out_ptr1 + (x5), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/33/c33le5u4y34t2heszci2oodf32sozih5jd5an32hjwjeyr3bwvv3.py
# Topologically Sorted Source Nodes: [input_441, low3_12, input_442, input_443], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_441 => convolution_146
#   input_442 => add_359, mul_454, mul_455, sub_147
#   input_443 => relu_147
#   low3_12 => add_357
# Graph fragment:
#   %convolution_146 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_146, %primals_882, %primals_883, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_357 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_146, %add_350), kwargs = {})
#   %sub_147 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_357, %unsqueeze_1180), kwargs = {})
#   %mul_454 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_147, %unsqueeze_1182), kwargs = {})
#   %mul_455 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_454, %unsqueeze_1184), kwargs = {})
#   %add_359 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_455, %unsqueeze_1186), kwargs = {})
#   %relu_147 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_359,), kwargs = {})
#   %sub_164 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_357, %unsqueeze_1350), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x3), tmp21, None)
    tl.store(out_ptr1 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/mw/cmwi2fzg5rtygfkgxkajb6va6wih7c4lnqbmfptjjj4obh3yfdru.py
# Topologically Sorted Source Nodes: [up2_3], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   up2_3 => add_379, add_380, convert_element_type_324, convert_element_type_325, iota_6, mul_480, mul_481
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_480 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_379 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_480, 0), kwargs = {})
#   %convert_element_type_324 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_379, torch.float32), kwargs = {})
#   %add_380 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_324, 0.0), kwargs = {})
#   %mul_481 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_380, 0.5), kwargs = {})
#   %convert_element_type_325 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_481, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_41 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_41(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: inductor_cache/nf/cnflsvqdddrdyjbmvc5owwamw7zoyavig4kxqy3ee2kdvuek5ive.py
# Topologically Sorted Source Nodes: [input_36, up1_3, input_459, low3_14, input_468, low3_15, up2_3, add_55], Original ATen: [aten.convolution, aten.add, aten._unsafe_index, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_55 => add_383
#   input_36 => convolution_11
#   input_459 => convolution_152
#   input_468 => convolution_155
#   low3_14 => add_371
#   low3_15 => add_378
#   up1_3 => add_27
#   up2_3 => _unsafe_index_3
# Graph fragment:
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_72, %primals_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_11, %add_20), kwargs = {})
#   %convolution_152 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_152, %primals_918, %primals_919, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_371 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_152, %add_364), kwargs = {})
#   %convolution_155 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_155, %primals_936, %primals_937, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_378 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_155, %add_371), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_378, [None, None, %unsqueeze_1251, %convert_element_type_325]), kwargs = {})
#   %add_383 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %_unsafe_index_3), kwargs = {})
#   %sub_302 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_20, %unsqueeze_3006), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_native_batch_norm_backward_42 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_native_batch_norm_backward_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_native_batch_norm_backward_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_native_batch_norm_backward_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x2 = ((xindex // 4096) % 4)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x6 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x5), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tl.full([XBLOCK], 32, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tmp11 = tmp10 + tmp6
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr3 + (tmp13 + 32*tmp9 + 1024*x6), None, eviction_policy='evict_last')
    tmp16 = tmp14 + tmp15
    tmp17 = tl.load(in_ptr5 + (tmp13 + 32*tmp9 + 1024*x6), None, eviction_policy='evict_last')
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr7 + (tmp13 + 32*tmp9 + 1024*x6), None, eviction_policy='evict_last')
    tmp21 = tmp19 + tmp20
    tmp22 = tmp16 + tmp21
    tmp23 = tmp4 + tmp22
    tmp25 = tmp3 - tmp24
    tl.store(in_out_ptr0 + (x5), tmp23, None)
    tl.store(out_ptr0 + (x5), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/o2/co2zjmz76zwigwsqniah35v7n2jzrtmf2wzxn4672hheufgtypof.py
# Topologically Sorted Source Nodes: [input_387, low3_6], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_387 => convolution_128
#   low3_6 => add_305
# Graph fragment:
#   %convolution_128 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_128, %primals_774, %primals_775, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_305 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_128, %add_298), kwargs = {})
#   %sub_182 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_305, %unsqueeze_1566), kwargs = {})
#   %sub_185 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_298, %unsqueeze_1602), kwargs = {})
triton_poi_fused_add_convolution_native_batch_norm_backward_43 = async_compile.triton('triton_poi_fused_add_convolution_native_batch_norm_backward_43', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_batch_norm_backward_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_batch_norm_backward_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tmp3 - tmp7
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ol/colvutwwtnmwtzux25uphvnrrnwirvharqlqqt2u2ztz3twwt6mj.py
# Topologically Sorted Source Nodes: [input_342, low3_1], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_342 => convolution_113
#   low3_1 => add_265
# Graph fragment:
#   %convolution_113 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_113, %primals_684, %primals_685, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_265 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_113, %add_258), kwargs = {})
#   %sub_197 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_265, %unsqueeze_1746), kwargs = {})
#   %sub_200 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_258, %unsqueeze_1782), kwargs = {})
triton_poi_fused_add_convolution_native_batch_norm_backward_44 = async_compile.triton('triton_poi_fused_add_convolution_native_batch_norm_backward_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_batch_norm_backward_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_batch_norm_backward_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tmp3 - tmp7
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mu/cmugetdkbgauxzw556zbfoybuzlkzxkux5gi6nofoifun6r5t7hj.py
# Topologically Sorted Source Nodes: [low1_15], Original ATen: [aten.max_pool2d_with_indices, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   low1_15 => _low_memory_max_pool2d_with_offsets_3
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_167, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %sub_227 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_6, %unsqueeze_2106), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_45 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_45(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x4 = xindex // 4
    x2 = ((xindex // 16) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = tmp6 - tmp7
    tl.store(out_ptr0 + (x5), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/af/cafoarvmf2ysq7dglecpadk4q5o7bzkvfv7yk65wwt3v5ehoyfqy.py
# Topologically Sorted Source Nodes: [input_225, up1_12], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_225 => convolution_74
#   up1_12 => add_174
# Graph fragment:
#   %convolution_74 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_74, %primals_450, %primals_451, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_174 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_74, %add_167), kwargs = {})
#   %sub_236 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_174, %unsqueeze_2214), kwargs = {})
triton_poi_fused_add_convolution_native_batch_norm_backward_46 = async_compile.triton('triton_poi_fused_add_convolution_native_batch_norm_backward_46', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_batch_norm_backward_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_batch_norm_backward_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j4/cj4xy6zlawc6tcrqbb6gquvy2ffylxcbkilamfoms3i4cojhfyw3.py
# Topologically Sorted Source Nodes: [low1_10], Original ATen: [aten.max_pool2d_with_indices, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   low1_10 => _low_memory_max_pool2d_with_offsets_2
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_111, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %sub_251 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_4, %unsqueeze_2394), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_47 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_47(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x4 = xindex // 8
    x2 = ((xindex // 64) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = tmp6 - tmp7
    tl.store(out_ptr0 + (x5), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5m/c5msdvueac4lxx4ybv6vvtildwt3kwysisbkmde2duaregbilcxa.py
# Topologically Sorted Source Nodes: [input_171, up1_10], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_171 => convolution_56
#   up1_10 => add_132
# Graph fragment:
#   %convolution_56 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_56, %primals_342, %primals_343, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_132 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_56, %add_125), kwargs = {})
#   %sub_254 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_132, %unsqueeze_2430), kwargs = {})
#   %sub_257 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_125, %unsqueeze_2466), kwargs = {})
triton_poi_fused_add_convolution_native_batch_norm_backward_48 = async_compile.triton('triton_poi_fused_add_convolution_native_batch_norm_backward_48', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_batch_norm_backward_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_batch_norm_backward_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tmp3 - tmp7
    tl.store(in_out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/dg/cdg7rk5mbup447t4t643pluxdr3fzx6zjgacalwof3ucgytpshy4.py
# Topologically Sorted Source Nodes: [input_153, up1_8], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_153 => convolution_50
#   up1_8 => add_118
# Graph fragment:
#   %convolution_50 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_50, %primals_306, %primals_307, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_118 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_50, %add_111), kwargs = {})
#   %sub_260 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_118, %unsqueeze_2502), kwargs = {})
triton_poi_fused_add_convolution_native_batch_norm_backward_49 = async_compile.triton('triton_poi_fused_add_convolution_native_batch_norm_backward_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_batch_norm_backward_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_batch_norm_backward_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/va/cvabfqwb4oww7wawvmdp5k4tp44sxzh7hh47ffny3mhzobuy2wu3.py
# Topologically Sorted Source Nodes: [low1_5], Original ATen: [aten.max_pool2d_with_indices, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   low1_5 => _low_memory_max_pool2d_with_offsets_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_55, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %sub_275 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_2, %unsqueeze_2682), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_50 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_50(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x4 = xindex // 16
    x2 = ((xindex // 256) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x4), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = tmp6 - tmp7
    tl.store(out_ptr0 + (x5), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/ww/cwwqwkn7o37py5ba5tcpowovjiqolxsi2fteoa2lkpuvqcusx3wl.py
# Topologically Sorted Source Nodes: [input_99, up1_6], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_99 => convolution_32
#   up1_6 => add_76
# Graph fragment:
#   %convolution_32 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_32, %primals_198, %primals_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_76 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_32, %add_69), kwargs = {})
#   %sub_278 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_76, %unsqueeze_2718), kwargs = {})
#   %sub_281 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_69, %unsqueeze_2754), kwargs = {})
triton_poi_fused_add_convolution_native_batch_norm_backward_51 = async_compile.triton('triton_poi_fused_add_convolution_native_batch_norm_backward_51', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_batch_norm_backward_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_batch_norm_backward_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tmp3 - tmp7
    tl.store(in_out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/3x/c3xodffo2l63ptxedc7su6v2gduczrif3g3q3dr3axdzjd56niuf.py
# Topologically Sorted Source Nodes: [input_81, up1_4], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_81 => convolution_26
#   up1_4 => add_62
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_26, %primals_162, %primals_163, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_62 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %add_55), kwargs = {})
#   %sub_284 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_62, %unsqueeze_2790), kwargs = {})
triton_poi_fused_add_convolution_native_batch_norm_backward_52 = async_compile.triton('triton_poi_fused_add_convolution_native_batch_norm_backward_52', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_batch_norm_backward_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_batch_norm_backward_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/vg/cvgxeqtwxlnvhe7cfaiupshczjh4hzykdwskixkyrfeqtbusc2id.py
# Topologically Sorted Source Nodes: [low1], Original ATen: [aten.max_pool2d_with_indices, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   low1 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%primals_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %sub_299 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_2970), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_53 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_53', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_53(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x4 = xindex // 32
    x2 = ((xindex // 1024) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = tmp6 - tmp7
    tl.store(out_ptr0 + (x5), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/k3/ck37nrtyzacyb3tzr4pqyt4buvpoeb4zyjsump7yi5qwczlzhlwx.py
# Topologically Sorted Source Nodes: [up1, input_18, up1_1], Original ATen: [aten.add, aten.convolution, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_18 => convolution_5
#   up1 => add_6
#   up1_1 => add_13
# Graph fragment:
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %primals_1), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_36, %primals_37, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add_6), kwargs = {})
#   %sub_305 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %unsqueeze_3042), kwargs = {})
triton_poi_fused_add_convolution_native_batch_norm_backward_54 = async_compile.triton('triton_poi_fused_add_convolution_native_batch_norm_backward_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_batch_norm_backward_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_batch_norm_backward_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_7, (2, ), (1, ))
    assert_size_stride(primals_8, (2, ), (1, ))
    assert_size_stride(primals_9, (2, ), (1, ))
    assert_size_stride(primals_10, (2, ), (1, ))
    assert_size_stride(primals_11, (2, ), (1, ))
    assert_size_stride(primals_12, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_13, (2, ), (1, ))
    assert_size_stride(primals_14, (2, ), (1, ))
    assert_size_stride(primals_15, (2, ), (1, ))
    assert_size_stride(primals_16, (2, ), (1, ))
    assert_size_stride(primals_17, (2, ), (1, ))
    assert_size_stride(primals_18, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, ), (1, ))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_25, (2, ), (1, ))
    assert_size_stride(primals_26, (2, ), (1, ))
    assert_size_stride(primals_27, (2, ), (1, ))
    assert_size_stride(primals_28, (2, ), (1, ))
    assert_size_stride(primals_29, (2, ), (1, ))
    assert_size_stride(primals_30, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_31, (2, ), (1, ))
    assert_size_stride(primals_32, (2, ), (1, ))
    assert_size_stride(primals_33, (2, ), (1, ))
    assert_size_stride(primals_34, (2, ), (1, ))
    assert_size_stride(primals_35, (2, ), (1, ))
    assert_size_stride(primals_36, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_37, (4, ), (1, ))
    assert_size_stride(primals_38, (4, ), (1, ))
    assert_size_stride(primals_39, (4, ), (1, ))
    assert_size_stride(primals_40, (4, ), (1, ))
    assert_size_stride(primals_41, (4, ), (1, ))
    assert_size_stride(primals_42, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_43, (2, ), (1, ))
    assert_size_stride(primals_44, (2, ), (1, ))
    assert_size_stride(primals_45, (2, ), (1, ))
    assert_size_stride(primals_46, (2, ), (1, ))
    assert_size_stride(primals_47, (2, ), (1, ))
    assert_size_stride(primals_48, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_49, (2, ), (1, ))
    assert_size_stride(primals_50, (2, ), (1, ))
    assert_size_stride(primals_51, (2, ), (1, ))
    assert_size_stride(primals_52, (2, ), (1, ))
    assert_size_stride(primals_53, (2, ), (1, ))
    assert_size_stride(primals_54, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_55, (4, ), (1, ))
    assert_size_stride(primals_56, (4, ), (1, ))
    assert_size_stride(primals_57, (4, ), (1, ))
    assert_size_stride(primals_58, (4, ), (1, ))
    assert_size_stride(primals_59, (4, ), (1, ))
    assert_size_stride(primals_60, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_61, (2, ), (1, ))
    assert_size_stride(primals_62, (2, ), (1, ))
    assert_size_stride(primals_63, (2, ), (1, ))
    assert_size_stride(primals_64, (2, ), (1, ))
    assert_size_stride(primals_65, (2, ), (1, ))
    assert_size_stride(primals_66, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_67, (2, ), (1, ))
    assert_size_stride(primals_68, (2, ), (1, ))
    assert_size_stride(primals_69, (2, ), (1, ))
    assert_size_stride(primals_70, (2, ), (1, ))
    assert_size_stride(primals_71, (2, ), (1, ))
    assert_size_stride(primals_72, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_73, (4, ), (1, ))
    assert_size_stride(primals_74, (4, ), (1, ))
    assert_size_stride(primals_75, (4, ), (1, ))
    assert_size_stride(primals_76, (4, ), (1, ))
    assert_size_stride(primals_77, (4, ), (1, ))
    assert_size_stride(primals_78, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_79, (2, ), (1, ))
    assert_size_stride(primals_80, (2, ), (1, ))
    assert_size_stride(primals_81, (2, ), (1, ))
    assert_size_stride(primals_82, (2, ), (1, ))
    assert_size_stride(primals_83, (2, ), (1, ))
    assert_size_stride(primals_84, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_85, (2, ), (1, ))
    assert_size_stride(primals_86, (2, ), (1, ))
    assert_size_stride(primals_87, (2, ), (1, ))
    assert_size_stride(primals_88, (2, ), (1, ))
    assert_size_stride(primals_89, (2, ), (1, ))
    assert_size_stride(primals_90, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_91, (4, ), (1, ))
    assert_size_stride(primals_92, (4, ), (1, ))
    assert_size_stride(primals_93, (4, ), (1, ))
    assert_size_stride(primals_94, (4, ), (1, ))
    assert_size_stride(primals_95, (4, ), (1, ))
    assert_size_stride(primals_96, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_97, (2, ), (1, ))
    assert_size_stride(primals_98, (2, ), (1, ))
    assert_size_stride(primals_99, (2, ), (1, ))
    assert_size_stride(primals_100, (2, ), (1, ))
    assert_size_stride(primals_101, (2, ), (1, ))
    assert_size_stride(primals_102, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_103, (2, ), (1, ))
    assert_size_stride(primals_104, (2, ), (1, ))
    assert_size_stride(primals_105, (2, ), (1, ))
    assert_size_stride(primals_106, (2, ), (1, ))
    assert_size_stride(primals_107, (2, ), (1, ))
    assert_size_stride(primals_108, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_109, (4, ), (1, ))
    assert_size_stride(primals_110, (4, ), (1, ))
    assert_size_stride(primals_111, (4, ), (1, ))
    assert_size_stride(primals_112, (4, ), (1, ))
    assert_size_stride(primals_113, (4, ), (1, ))
    assert_size_stride(primals_114, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_115, (2, ), (1, ))
    assert_size_stride(primals_116, (2, ), (1, ))
    assert_size_stride(primals_117, (2, ), (1, ))
    assert_size_stride(primals_118, (2, ), (1, ))
    assert_size_stride(primals_119, (2, ), (1, ))
    assert_size_stride(primals_120, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_121, (2, ), (1, ))
    assert_size_stride(primals_122, (2, ), (1, ))
    assert_size_stride(primals_123, (2, ), (1, ))
    assert_size_stride(primals_124, (2, ), (1, ))
    assert_size_stride(primals_125, (2, ), (1, ))
    assert_size_stride(primals_126, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_127, (4, ), (1, ))
    assert_size_stride(primals_128, (4, ), (1, ))
    assert_size_stride(primals_129, (4, ), (1, ))
    assert_size_stride(primals_130, (4, ), (1, ))
    assert_size_stride(primals_131, (4, ), (1, ))
    assert_size_stride(primals_132, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_133, (2, ), (1, ))
    assert_size_stride(primals_134, (2, ), (1, ))
    assert_size_stride(primals_135, (2, ), (1, ))
    assert_size_stride(primals_136, (2, ), (1, ))
    assert_size_stride(primals_137, (2, ), (1, ))
    assert_size_stride(primals_138, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_139, (2, ), (1, ))
    assert_size_stride(primals_140, (2, ), (1, ))
    assert_size_stride(primals_141, (2, ), (1, ))
    assert_size_stride(primals_142, (2, ), (1, ))
    assert_size_stride(primals_143, (2, ), (1, ))
    assert_size_stride(primals_144, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_145, (4, ), (1, ))
    assert_size_stride(primals_146, (4, ), (1, ))
    assert_size_stride(primals_147, (4, ), (1, ))
    assert_size_stride(primals_148, (4, ), (1, ))
    assert_size_stride(primals_149, (4, ), (1, ))
    assert_size_stride(primals_150, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_151, (2, ), (1, ))
    assert_size_stride(primals_152, (2, ), (1, ))
    assert_size_stride(primals_153, (2, ), (1, ))
    assert_size_stride(primals_154, (2, ), (1, ))
    assert_size_stride(primals_155, (2, ), (1, ))
    assert_size_stride(primals_156, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_157, (2, ), (1, ))
    assert_size_stride(primals_158, (2, ), (1, ))
    assert_size_stride(primals_159, (2, ), (1, ))
    assert_size_stride(primals_160, (2, ), (1, ))
    assert_size_stride(primals_161, (2, ), (1, ))
    assert_size_stride(primals_162, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_163, (4, ), (1, ))
    assert_size_stride(primals_164, (4, ), (1, ))
    assert_size_stride(primals_165, (4, ), (1, ))
    assert_size_stride(primals_166, (4, ), (1, ))
    assert_size_stride(primals_167, (4, ), (1, ))
    assert_size_stride(primals_168, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_169, (2, ), (1, ))
    assert_size_stride(primals_170, (2, ), (1, ))
    assert_size_stride(primals_171, (2, ), (1, ))
    assert_size_stride(primals_172, (2, ), (1, ))
    assert_size_stride(primals_173, (2, ), (1, ))
    assert_size_stride(primals_174, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_175, (2, ), (1, ))
    assert_size_stride(primals_176, (2, ), (1, ))
    assert_size_stride(primals_177, (2, ), (1, ))
    assert_size_stride(primals_178, (2, ), (1, ))
    assert_size_stride(primals_179, (2, ), (1, ))
    assert_size_stride(primals_180, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_181, (4, ), (1, ))
    assert_size_stride(primals_182, (4, ), (1, ))
    assert_size_stride(primals_183, (4, ), (1, ))
    assert_size_stride(primals_184, (4, ), (1, ))
    assert_size_stride(primals_185, (4, ), (1, ))
    assert_size_stride(primals_186, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_187, (2, ), (1, ))
    assert_size_stride(primals_188, (2, ), (1, ))
    assert_size_stride(primals_189, (2, ), (1, ))
    assert_size_stride(primals_190, (2, ), (1, ))
    assert_size_stride(primals_191, (2, ), (1, ))
    assert_size_stride(primals_192, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_193, (2, ), (1, ))
    assert_size_stride(primals_194, (2, ), (1, ))
    assert_size_stride(primals_195, (2, ), (1, ))
    assert_size_stride(primals_196, (2, ), (1, ))
    assert_size_stride(primals_197, (2, ), (1, ))
    assert_size_stride(primals_198, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_199, (4, ), (1, ))
    assert_size_stride(primals_200, (4, ), (1, ))
    assert_size_stride(primals_201, (4, ), (1, ))
    assert_size_stride(primals_202, (4, ), (1, ))
    assert_size_stride(primals_203, (4, ), (1, ))
    assert_size_stride(primals_204, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_205, (2, ), (1, ))
    assert_size_stride(primals_206, (2, ), (1, ))
    assert_size_stride(primals_207, (2, ), (1, ))
    assert_size_stride(primals_208, (2, ), (1, ))
    assert_size_stride(primals_209, (2, ), (1, ))
    assert_size_stride(primals_210, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_211, (2, ), (1, ))
    assert_size_stride(primals_212, (2, ), (1, ))
    assert_size_stride(primals_213, (2, ), (1, ))
    assert_size_stride(primals_214, (2, ), (1, ))
    assert_size_stride(primals_215, (2, ), (1, ))
    assert_size_stride(primals_216, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_217, (4, ), (1, ))
    assert_size_stride(primals_218, (4, ), (1, ))
    assert_size_stride(primals_219, (4, ), (1, ))
    assert_size_stride(primals_220, (4, ), (1, ))
    assert_size_stride(primals_221, (4, ), (1, ))
    assert_size_stride(primals_222, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_223, (2, ), (1, ))
    assert_size_stride(primals_224, (2, ), (1, ))
    assert_size_stride(primals_225, (2, ), (1, ))
    assert_size_stride(primals_226, (2, ), (1, ))
    assert_size_stride(primals_227, (2, ), (1, ))
    assert_size_stride(primals_228, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_229, (2, ), (1, ))
    assert_size_stride(primals_230, (2, ), (1, ))
    assert_size_stride(primals_231, (2, ), (1, ))
    assert_size_stride(primals_232, (2, ), (1, ))
    assert_size_stride(primals_233, (2, ), (1, ))
    assert_size_stride(primals_234, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_235, (4, ), (1, ))
    assert_size_stride(primals_236, (4, ), (1, ))
    assert_size_stride(primals_237, (4, ), (1, ))
    assert_size_stride(primals_238, (4, ), (1, ))
    assert_size_stride(primals_239, (4, ), (1, ))
    assert_size_stride(primals_240, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_241, (2, ), (1, ))
    assert_size_stride(primals_242, (2, ), (1, ))
    assert_size_stride(primals_243, (2, ), (1, ))
    assert_size_stride(primals_244, (2, ), (1, ))
    assert_size_stride(primals_245, (2, ), (1, ))
    assert_size_stride(primals_246, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_247, (2, ), (1, ))
    assert_size_stride(primals_248, (2, ), (1, ))
    assert_size_stride(primals_249, (2, ), (1, ))
    assert_size_stride(primals_250, (2, ), (1, ))
    assert_size_stride(primals_251, (2, ), (1, ))
    assert_size_stride(primals_252, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_253, (4, ), (1, ))
    assert_size_stride(primals_254, (4, ), (1, ))
    assert_size_stride(primals_255, (4, ), (1, ))
    assert_size_stride(primals_256, (4, ), (1, ))
    assert_size_stride(primals_257, (4, ), (1, ))
    assert_size_stride(primals_258, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_259, (2, ), (1, ))
    assert_size_stride(primals_260, (2, ), (1, ))
    assert_size_stride(primals_261, (2, ), (1, ))
    assert_size_stride(primals_262, (2, ), (1, ))
    assert_size_stride(primals_263, (2, ), (1, ))
    assert_size_stride(primals_264, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_265, (2, ), (1, ))
    assert_size_stride(primals_266, (2, ), (1, ))
    assert_size_stride(primals_267, (2, ), (1, ))
    assert_size_stride(primals_268, (2, ), (1, ))
    assert_size_stride(primals_269, (2, ), (1, ))
    assert_size_stride(primals_270, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_271, (4, ), (1, ))
    assert_size_stride(primals_272, (4, ), (1, ))
    assert_size_stride(primals_273, (4, ), (1, ))
    assert_size_stride(primals_274, (4, ), (1, ))
    assert_size_stride(primals_275, (4, ), (1, ))
    assert_size_stride(primals_276, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_277, (2, ), (1, ))
    assert_size_stride(primals_278, (2, ), (1, ))
    assert_size_stride(primals_279, (2, ), (1, ))
    assert_size_stride(primals_280, (2, ), (1, ))
    assert_size_stride(primals_281, (2, ), (1, ))
    assert_size_stride(primals_282, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_283, (2, ), (1, ))
    assert_size_stride(primals_284, (2, ), (1, ))
    assert_size_stride(primals_285, (2, ), (1, ))
    assert_size_stride(primals_286, (2, ), (1, ))
    assert_size_stride(primals_287, (2, ), (1, ))
    assert_size_stride(primals_288, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_289, (4, ), (1, ))
    assert_size_stride(primals_290, (4, ), (1, ))
    assert_size_stride(primals_291, (4, ), (1, ))
    assert_size_stride(primals_292, (4, ), (1, ))
    assert_size_stride(primals_293, (4, ), (1, ))
    assert_size_stride(primals_294, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_295, (2, ), (1, ))
    assert_size_stride(primals_296, (2, ), (1, ))
    assert_size_stride(primals_297, (2, ), (1, ))
    assert_size_stride(primals_298, (2, ), (1, ))
    assert_size_stride(primals_299, (2, ), (1, ))
    assert_size_stride(primals_300, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_301, (2, ), (1, ))
    assert_size_stride(primals_302, (2, ), (1, ))
    assert_size_stride(primals_303, (2, ), (1, ))
    assert_size_stride(primals_304, (2, ), (1, ))
    assert_size_stride(primals_305, (2, ), (1, ))
    assert_size_stride(primals_306, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_307, (4, ), (1, ))
    assert_size_stride(primals_308, (4, ), (1, ))
    assert_size_stride(primals_309, (4, ), (1, ))
    assert_size_stride(primals_310, (4, ), (1, ))
    assert_size_stride(primals_311, (4, ), (1, ))
    assert_size_stride(primals_312, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_313, (2, ), (1, ))
    assert_size_stride(primals_314, (2, ), (1, ))
    assert_size_stride(primals_315, (2, ), (1, ))
    assert_size_stride(primals_316, (2, ), (1, ))
    assert_size_stride(primals_317, (2, ), (1, ))
    assert_size_stride(primals_318, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_319, (2, ), (1, ))
    assert_size_stride(primals_320, (2, ), (1, ))
    assert_size_stride(primals_321, (2, ), (1, ))
    assert_size_stride(primals_322, (2, ), (1, ))
    assert_size_stride(primals_323, (2, ), (1, ))
    assert_size_stride(primals_324, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_325, (4, ), (1, ))
    assert_size_stride(primals_326, (4, ), (1, ))
    assert_size_stride(primals_327, (4, ), (1, ))
    assert_size_stride(primals_328, (4, ), (1, ))
    assert_size_stride(primals_329, (4, ), (1, ))
    assert_size_stride(primals_330, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_331, (2, ), (1, ))
    assert_size_stride(primals_332, (2, ), (1, ))
    assert_size_stride(primals_333, (2, ), (1, ))
    assert_size_stride(primals_334, (2, ), (1, ))
    assert_size_stride(primals_335, (2, ), (1, ))
    assert_size_stride(primals_336, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_337, (2, ), (1, ))
    assert_size_stride(primals_338, (2, ), (1, ))
    assert_size_stride(primals_339, (2, ), (1, ))
    assert_size_stride(primals_340, (2, ), (1, ))
    assert_size_stride(primals_341, (2, ), (1, ))
    assert_size_stride(primals_342, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_343, (4, ), (1, ))
    assert_size_stride(primals_344, (4, ), (1, ))
    assert_size_stride(primals_345, (4, ), (1, ))
    assert_size_stride(primals_346, (4, ), (1, ))
    assert_size_stride(primals_347, (4, ), (1, ))
    assert_size_stride(primals_348, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_349, (2, ), (1, ))
    assert_size_stride(primals_350, (2, ), (1, ))
    assert_size_stride(primals_351, (2, ), (1, ))
    assert_size_stride(primals_352, (2, ), (1, ))
    assert_size_stride(primals_353, (2, ), (1, ))
    assert_size_stride(primals_354, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_355, (2, ), (1, ))
    assert_size_stride(primals_356, (2, ), (1, ))
    assert_size_stride(primals_357, (2, ), (1, ))
    assert_size_stride(primals_358, (2, ), (1, ))
    assert_size_stride(primals_359, (2, ), (1, ))
    assert_size_stride(primals_360, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_361, (4, ), (1, ))
    assert_size_stride(primals_362, (4, ), (1, ))
    assert_size_stride(primals_363, (4, ), (1, ))
    assert_size_stride(primals_364, (4, ), (1, ))
    assert_size_stride(primals_365, (4, ), (1, ))
    assert_size_stride(primals_366, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_367, (2, ), (1, ))
    assert_size_stride(primals_368, (2, ), (1, ))
    assert_size_stride(primals_369, (2, ), (1, ))
    assert_size_stride(primals_370, (2, ), (1, ))
    assert_size_stride(primals_371, (2, ), (1, ))
    assert_size_stride(primals_372, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_373, (2, ), (1, ))
    assert_size_stride(primals_374, (2, ), (1, ))
    assert_size_stride(primals_375, (2, ), (1, ))
    assert_size_stride(primals_376, (2, ), (1, ))
    assert_size_stride(primals_377, (2, ), (1, ))
    assert_size_stride(primals_378, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_379, (4, ), (1, ))
    assert_size_stride(primals_380, (4, ), (1, ))
    assert_size_stride(primals_381, (4, ), (1, ))
    assert_size_stride(primals_382, (4, ), (1, ))
    assert_size_stride(primals_383, (4, ), (1, ))
    assert_size_stride(primals_384, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_385, (2, ), (1, ))
    assert_size_stride(primals_386, (2, ), (1, ))
    assert_size_stride(primals_387, (2, ), (1, ))
    assert_size_stride(primals_388, (2, ), (1, ))
    assert_size_stride(primals_389, (2, ), (1, ))
    assert_size_stride(primals_390, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_391, (2, ), (1, ))
    assert_size_stride(primals_392, (2, ), (1, ))
    assert_size_stride(primals_393, (2, ), (1, ))
    assert_size_stride(primals_394, (2, ), (1, ))
    assert_size_stride(primals_395, (2, ), (1, ))
    assert_size_stride(primals_396, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_397, (4, ), (1, ))
    assert_size_stride(primals_398, (4, ), (1, ))
    assert_size_stride(primals_399, (4, ), (1, ))
    assert_size_stride(primals_400, (4, ), (1, ))
    assert_size_stride(primals_401, (4, ), (1, ))
    assert_size_stride(primals_402, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_403, (2, ), (1, ))
    assert_size_stride(primals_404, (2, ), (1, ))
    assert_size_stride(primals_405, (2, ), (1, ))
    assert_size_stride(primals_406, (2, ), (1, ))
    assert_size_stride(primals_407, (2, ), (1, ))
    assert_size_stride(primals_408, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_409, (2, ), (1, ))
    assert_size_stride(primals_410, (2, ), (1, ))
    assert_size_stride(primals_411, (2, ), (1, ))
    assert_size_stride(primals_412, (2, ), (1, ))
    assert_size_stride(primals_413, (2, ), (1, ))
    assert_size_stride(primals_414, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_415, (4, ), (1, ))
    assert_size_stride(primals_416, (4, ), (1, ))
    assert_size_stride(primals_417, (4, ), (1, ))
    assert_size_stride(primals_418, (4, ), (1, ))
    assert_size_stride(primals_419, (4, ), (1, ))
    assert_size_stride(primals_420, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_421, (2, ), (1, ))
    assert_size_stride(primals_422, (2, ), (1, ))
    assert_size_stride(primals_423, (2, ), (1, ))
    assert_size_stride(primals_424, (2, ), (1, ))
    assert_size_stride(primals_425, (2, ), (1, ))
    assert_size_stride(primals_426, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_427, (2, ), (1, ))
    assert_size_stride(primals_428, (2, ), (1, ))
    assert_size_stride(primals_429, (2, ), (1, ))
    assert_size_stride(primals_430, (2, ), (1, ))
    assert_size_stride(primals_431, (2, ), (1, ))
    assert_size_stride(primals_432, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_433, (4, ), (1, ))
    assert_size_stride(primals_434, (4, ), (1, ))
    assert_size_stride(primals_435, (4, ), (1, ))
    assert_size_stride(primals_436, (4, ), (1, ))
    assert_size_stride(primals_437, (4, ), (1, ))
    assert_size_stride(primals_438, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_439, (2, ), (1, ))
    assert_size_stride(primals_440, (2, ), (1, ))
    assert_size_stride(primals_441, (2, ), (1, ))
    assert_size_stride(primals_442, (2, ), (1, ))
    assert_size_stride(primals_443, (2, ), (1, ))
    assert_size_stride(primals_444, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_445, (2, ), (1, ))
    assert_size_stride(primals_446, (2, ), (1, ))
    assert_size_stride(primals_447, (2, ), (1, ))
    assert_size_stride(primals_448, (2, ), (1, ))
    assert_size_stride(primals_449, (2, ), (1, ))
    assert_size_stride(primals_450, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_451, (4, ), (1, ))
    assert_size_stride(primals_452, (4, ), (1, ))
    assert_size_stride(primals_453, (4, ), (1, ))
    assert_size_stride(primals_454, (4, ), (1, ))
    assert_size_stride(primals_455, (4, ), (1, ))
    assert_size_stride(primals_456, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_457, (2, ), (1, ))
    assert_size_stride(primals_458, (2, ), (1, ))
    assert_size_stride(primals_459, (2, ), (1, ))
    assert_size_stride(primals_460, (2, ), (1, ))
    assert_size_stride(primals_461, (2, ), (1, ))
    assert_size_stride(primals_462, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_463, (2, ), (1, ))
    assert_size_stride(primals_464, (2, ), (1, ))
    assert_size_stride(primals_465, (2, ), (1, ))
    assert_size_stride(primals_466, (2, ), (1, ))
    assert_size_stride(primals_467, (2, ), (1, ))
    assert_size_stride(primals_468, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_469, (4, ), (1, ))
    assert_size_stride(primals_470, (4, ), (1, ))
    assert_size_stride(primals_471, (4, ), (1, ))
    assert_size_stride(primals_472, (4, ), (1, ))
    assert_size_stride(primals_473, (4, ), (1, ))
    assert_size_stride(primals_474, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_475, (2, ), (1, ))
    assert_size_stride(primals_476, (2, ), (1, ))
    assert_size_stride(primals_477, (2, ), (1, ))
    assert_size_stride(primals_478, (2, ), (1, ))
    assert_size_stride(primals_479, (2, ), (1, ))
    assert_size_stride(primals_480, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_481, (2, ), (1, ))
    assert_size_stride(primals_482, (2, ), (1, ))
    assert_size_stride(primals_483, (2, ), (1, ))
    assert_size_stride(primals_484, (2, ), (1, ))
    assert_size_stride(primals_485, (2, ), (1, ))
    assert_size_stride(primals_486, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_487, (4, ), (1, ))
    assert_size_stride(primals_488, (4, ), (1, ))
    assert_size_stride(primals_489, (4, ), (1, ))
    assert_size_stride(primals_490, (4, ), (1, ))
    assert_size_stride(primals_491, (4, ), (1, ))
    assert_size_stride(primals_492, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_493, (2, ), (1, ))
    assert_size_stride(primals_494, (2, ), (1, ))
    assert_size_stride(primals_495, (2, ), (1, ))
    assert_size_stride(primals_496, (2, ), (1, ))
    assert_size_stride(primals_497, (2, ), (1, ))
    assert_size_stride(primals_498, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_499, (2, ), (1, ))
    assert_size_stride(primals_500, (2, ), (1, ))
    assert_size_stride(primals_501, (2, ), (1, ))
    assert_size_stride(primals_502, (2, ), (1, ))
    assert_size_stride(primals_503, (2, ), (1, ))
    assert_size_stride(primals_504, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_505, (4, ), (1, ))
    assert_size_stride(primals_506, (4, ), (1, ))
    assert_size_stride(primals_507, (4, ), (1, ))
    assert_size_stride(primals_508, (4, ), (1, ))
    assert_size_stride(primals_509, (4, ), (1, ))
    assert_size_stride(primals_510, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_511, (2, ), (1, ))
    assert_size_stride(primals_512, (2, ), (1, ))
    assert_size_stride(primals_513, (2, ), (1, ))
    assert_size_stride(primals_514, (2, ), (1, ))
    assert_size_stride(primals_515, (2, ), (1, ))
    assert_size_stride(primals_516, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_517, (2, ), (1, ))
    assert_size_stride(primals_518, (2, ), (1, ))
    assert_size_stride(primals_519, (2, ), (1, ))
    assert_size_stride(primals_520, (2, ), (1, ))
    assert_size_stride(primals_521, (2, ), (1, ))
    assert_size_stride(primals_522, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_523, (4, ), (1, ))
    assert_size_stride(primals_524, (4, ), (1, ))
    assert_size_stride(primals_525, (4, ), (1, ))
    assert_size_stride(primals_526, (4, ), (1, ))
    assert_size_stride(primals_527, (4, ), (1, ))
    assert_size_stride(primals_528, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_529, (2, ), (1, ))
    assert_size_stride(primals_530, (2, ), (1, ))
    assert_size_stride(primals_531, (2, ), (1, ))
    assert_size_stride(primals_532, (2, ), (1, ))
    assert_size_stride(primals_533, (2, ), (1, ))
    assert_size_stride(primals_534, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_535, (2, ), (1, ))
    assert_size_stride(primals_536, (2, ), (1, ))
    assert_size_stride(primals_537, (2, ), (1, ))
    assert_size_stride(primals_538, (2, ), (1, ))
    assert_size_stride(primals_539, (2, ), (1, ))
    assert_size_stride(primals_540, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_541, (4, ), (1, ))
    assert_size_stride(primals_542, (4, ), (1, ))
    assert_size_stride(primals_543, (4, ), (1, ))
    assert_size_stride(primals_544, (4, ), (1, ))
    assert_size_stride(primals_545, (4, ), (1, ))
    assert_size_stride(primals_546, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_547, (2, ), (1, ))
    assert_size_stride(primals_548, (2, ), (1, ))
    assert_size_stride(primals_549, (2, ), (1, ))
    assert_size_stride(primals_550, (2, ), (1, ))
    assert_size_stride(primals_551, (2, ), (1, ))
    assert_size_stride(primals_552, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_553, (2, ), (1, ))
    assert_size_stride(primals_554, (2, ), (1, ))
    assert_size_stride(primals_555, (2, ), (1, ))
    assert_size_stride(primals_556, (2, ), (1, ))
    assert_size_stride(primals_557, (2, ), (1, ))
    assert_size_stride(primals_558, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_559, (4, ), (1, ))
    assert_size_stride(primals_560, (4, ), (1, ))
    assert_size_stride(primals_561, (4, ), (1, ))
    assert_size_stride(primals_562, (4, ), (1, ))
    assert_size_stride(primals_563, (4, ), (1, ))
    assert_size_stride(primals_564, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_565, (2, ), (1, ))
    assert_size_stride(primals_566, (2, ), (1, ))
    assert_size_stride(primals_567, (2, ), (1, ))
    assert_size_stride(primals_568, (2, ), (1, ))
    assert_size_stride(primals_569, (2, ), (1, ))
    assert_size_stride(primals_570, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_571, (2, ), (1, ))
    assert_size_stride(primals_572, (2, ), (1, ))
    assert_size_stride(primals_573, (2, ), (1, ))
    assert_size_stride(primals_574, (2, ), (1, ))
    assert_size_stride(primals_575, (2, ), (1, ))
    assert_size_stride(primals_576, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_577, (4, ), (1, ))
    assert_size_stride(primals_578, (4, ), (1, ))
    assert_size_stride(primals_579, (4, ), (1, ))
    assert_size_stride(primals_580, (4, ), (1, ))
    assert_size_stride(primals_581, (4, ), (1, ))
    assert_size_stride(primals_582, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_583, (2, ), (1, ))
    assert_size_stride(primals_584, (2, ), (1, ))
    assert_size_stride(primals_585, (2, ), (1, ))
    assert_size_stride(primals_586, (2, ), (1, ))
    assert_size_stride(primals_587, (2, ), (1, ))
    assert_size_stride(primals_588, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_589, (2, ), (1, ))
    assert_size_stride(primals_590, (2, ), (1, ))
    assert_size_stride(primals_591, (2, ), (1, ))
    assert_size_stride(primals_592, (2, ), (1, ))
    assert_size_stride(primals_593, (2, ), (1, ))
    assert_size_stride(primals_594, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_595, (4, ), (1, ))
    assert_size_stride(primals_596, (4, ), (1, ))
    assert_size_stride(primals_597, (4, ), (1, ))
    assert_size_stride(primals_598, (4, ), (1, ))
    assert_size_stride(primals_599, (4, ), (1, ))
    assert_size_stride(primals_600, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_601, (2, ), (1, ))
    assert_size_stride(primals_602, (2, ), (1, ))
    assert_size_stride(primals_603, (2, ), (1, ))
    assert_size_stride(primals_604, (2, ), (1, ))
    assert_size_stride(primals_605, (2, ), (1, ))
    assert_size_stride(primals_606, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_607, (2, ), (1, ))
    assert_size_stride(primals_608, (2, ), (1, ))
    assert_size_stride(primals_609, (2, ), (1, ))
    assert_size_stride(primals_610, (2, ), (1, ))
    assert_size_stride(primals_611, (2, ), (1, ))
    assert_size_stride(primals_612, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_613, (4, ), (1, ))
    assert_size_stride(primals_614, (4, ), (1, ))
    assert_size_stride(primals_615, (4, ), (1, ))
    assert_size_stride(primals_616, (4, ), (1, ))
    assert_size_stride(primals_617, (4, ), (1, ))
    assert_size_stride(primals_618, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_619, (2, ), (1, ))
    assert_size_stride(primals_620, (2, ), (1, ))
    assert_size_stride(primals_621, (2, ), (1, ))
    assert_size_stride(primals_622, (2, ), (1, ))
    assert_size_stride(primals_623, (2, ), (1, ))
    assert_size_stride(primals_624, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_625, (2, ), (1, ))
    assert_size_stride(primals_626, (2, ), (1, ))
    assert_size_stride(primals_627, (2, ), (1, ))
    assert_size_stride(primals_628, (2, ), (1, ))
    assert_size_stride(primals_629, (2, ), (1, ))
    assert_size_stride(primals_630, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_631, (4, ), (1, ))
    assert_size_stride(primals_632, (4, ), (1, ))
    assert_size_stride(primals_633, (4, ), (1, ))
    assert_size_stride(primals_634, (4, ), (1, ))
    assert_size_stride(primals_635, (4, ), (1, ))
    assert_size_stride(primals_636, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_637, (2, ), (1, ))
    assert_size_stride(primals_638, (2, ), (1, ))
    assert_size_stride(primals_639, (2, ), (1, ))
    assert_size_stride(primals_640, (2, ), (1, ))
    assert_size_stride(primals_641, (2, ), (1, ))
    assert_size_stride(primals_642, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_643, (2, ), (1, ))
    assert_size_stride(primals_644, (2, ), (1, ))
    assert_size_stride(primals_645, (2, ), (1, ))
    assert_size_stride(primals_646, (2, ), (1, ))
    assert_size_stride(primals_647, (2, ), (1, ))
    assert_size_stride(primals_648, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_649, (4, ), (1, ))
    assert_size_stride(primals_650, (4, ), (1, ))
    assert_size_stride(primals_651, (4, ), (1, ))
    assert_size_stride(primals_652, (4, ), (1, ))
    assert_size_stride(primals_653, (4, ), (1, ))
    assert_size_stride(primals_654, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_655, (2, ), (1, ))
    assert_size_stride(primals_656, (2, ), (1, ))
    assert_size_stride(primals_657, (2, ), (1, ))
    assert_size_stride(primals_658, (2, ), (1, ))
    assert_size_stride(primals_659, (2, ), (1, ))
    assert_size_stride(primals_660, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_661, (2, ), (1, ))
    assert_size_stride(primals_662, (2, ), (1, ))
    assert_size_stride(primals_663, (2, ), (1, ))
    assert_size_stride(primals_664, (2, ), (1, ))
    assert_size_stride(primals_665, (2, ), (1, ))
    assert_size_stride(primals_666, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_667, (4, ), (1, ))
    assert_size_stride(primals_668, (4, ), (1, ))
    assert_size_stride(primals_669, (4, ), (1, ))
    assert_size_stride(primals_670, (4, ), (1, ))
    assert_size_stride(primals_671, (4, ), (1, ))
    assert_size_stride(primals_672, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_673, (2, ), (1, ))
    assert_size_stride(primals_674, (2, ), (1, ))
    assert_size_stride(primals_675, (2, ), (1, ))
    assert_size_stride(primals_676, (2, ), (1, ))
    assert_size_stride(primals_677, (2, ), (1, ))
    assert_size_stride(primals_678, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_679, (2, ), (1, ))
    assert_size_stride(primals_680, (2, ), (1, ))
    assert_size_stride(primals_681, (2, ), (1, ))
    assert_size_stride(primals_682, (2, ), (1, ))
    assert_size_stride(primals_683, (2, ), (1, ))
    assert_size_stride(primals_684, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_685, (4, ), (1, ))
    assert_size_stride(primals_686, (4, ), (1, ))
    assert_size_stride(primals_687, (4, ), (1, ))
    assert_size_stride(primals_688, (4, ), (1, ))
    assert_size_stride(primals_689, (4, ), (1, ))
    assert_size_stride(primals_690, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_691, (2, ), (1, ))
    assert_size_stride(primals_692, (2, ), (1, ))
    assert_size_stride(primals_693, (2, ), (1, ))
    assert_size_stride(primals_694, (2, ), (1, ))
    assert_size_stride(primals_695, (2, ), (1, ))
    assert_size_stride(primals_696, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_697, (2, ), (1, ))
    assert_size_stride(primals_698, (2, ), (1, ))
    assert_size_stride(primals_699, (2, ), (1, ))
    assert_size_stride(primals_700, (2, ), (1, ))
    assert_size_stride(primals_701, (2, ), (1, ))
    assert_size_stride(primals_702, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_703, (4, ), (1, ))
    assert_size_stride(primals_704, (4, ), (1, ))
    assert_size_stride(primals_705, (4, ), (1, ))
    assert_size_stride(primals_706, (4, ), (1, ))
    assert_size_stride(primals_707, (4, ), (1, ))
    assert_size_stride(primals_708, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_709, (2, ), (1, ))
    assert_size_stride(primals_710, (2, ), (1, ))
    assert_size_stride(primals_711, (2, ), (1, ))
    assert_size_stride(primals_712, (2, ), (1, ))
    assert_size_stride(primals_713, (2, ), (1, ))
    assert_size_stride(primals_714, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_715, (2, ), (1, ))
    assert_size_stride(primals_716, (2, ), (1, ))
    assert_size_stride(primals_717, (2, ), (1, ))
    assert_size_stride(primals_718, (2, ), (1, ))
    assert_size_stride(primals_719, (2, ), (1, ))
    assert_size_stride(primals_720, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_721, (4, ), (1, ))
    assert_size_stride(primals_722, (4, ), (1, ))
    assert_size_stride(primals_723, (4, ), (1, ))
    assert_size_stride(primals_724, (4, ), (1, ))
    assert_size_stride(primals_725, (4, ), (1, ))
    assert_size_stride(primals_726, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_727, (2, ), (1, ))
    assert_size_stride(primals_728, (2, ), (1, ))
    assert_size_stride(primals_729, (2, ), (1, ))
    assert_size_stride(primals_730, (2, ), (1, ))
    assert_size_stride(primals_731, (2, ), (1, ))
    assert_size_stride(primals_732, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_733, (2, ), (1, ))
    assert_size_stride(primals_734, (2, ), (1, ))
    assert_size_stride(primals_735, (2, ), (1, ))
    assert_size_stride(primals_736, (2, ), (1, ))
    assert_size_stride(primals_737, (2, ), (1, ))
    assert_size_stride(primals_738, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_739, (4, ), (1, ))
    assert_size_stride(primals_740, (4, ), (1, ))
    assert_size_stride(primals_741, (4, ), (1, ))
    assert_size_stride(primals_742, (4, ), (1, ))
    assert_size_stride(primals_743, (4, ), (1, ))
    assert_size_stride(primals_744, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_745, (2, ), (1, ))
    assert_size_stride(primals_746, (2, ), (1, ))
    assert_size_stride(primals_747, (2, ), (1, ))
    assert_size_stride(primals_748, (2, ), (1, ))
    assert_size_stride(primals_749, (2, ), (1, ))
    assert_size_stride(primals_750, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_751, (2, ), (1, ))
    assert_size_stride(primals_752, (2, ), (1, ))
    assert_size_stride(primals_753, (2, ), (1, ))
    assert_size_stride(primals_754, (2, ), (1, ))
    assert_size_stride(primals_755, (2, ), (1, ))
    assert_size_stride(primals_756, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_757, (4, ), (1, ))
    assert_size_stride(primals_758, (4, ), (1, ))
    assert_size_stride(primals_759, (4, ), (1, ))
    assert_size_stride(primals_760, (4, ), (1, ))
    assert_size_stride(primals_761, (4, ), (1, ))
    assert_size_stride(primals_762, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_763, (2, ), (1, ))
    assert_size_stride(primals_764, (2, ), (1, ))
    assert_size_stride(primals_765, (2, ), (1, ))
    assert_size_stride(primals_766, (2, ), (1, ))
    assert_size_stride(primals_767, (2, ), (1, ))
    assert_size_stride(primals_768, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_769, (2, ), (1, ))
    assert_size_stride(primals_770, (2, ), (1, ))
    assert_size_stride(primals_771, (2, ), (1, ))
    assert_size_stride(primals_772, (2, ), (1, ))
    assert_size_stride(primals_773, (2, ), (1, ))
    assert_size_stride(primals_774, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_775, (4, ), (1, ))
    assert_size_stride(primals_776, (4, ), (1, ))
    assert_size_stride(primals_777, (4, ), (1, ))
    assert_size_stride(primals_778, (4, ), (1, ))
    assert_size_stride(primals_779, (4, ), (1, ))
    assert_size_stride(primals_780, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_781, (2, ), (1, ))
    assert_size_stride(primals_782, (2, ), (1, ))
    assert_size_stride(primals_783, (2, ), (1, ))
    assert_size_stride(primals_784, (2, ), (1, ))
    assert_size_stride(primals_785, (2, ), (1, ))
    assert_size_stride(primals_786, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_787, (2, ), (1, ))
    assert_size_stride(primals_788, (2, ), (1, ))
    assert_size_stride(primals_789, (2, ), (1, ))
    assert_size_stride(primals_790, (2, ), (1, ))
    assert_size_stride(primals_791, (2, ), (1, ))
    assert_size_stride(primals_792, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_793, (4, ), (1, ))
    assert_size_stride(primals_794, (4, ), (1, ))
    assert_size_stride(primals_795, (4, ), (1, ))
    assert_size_stride(primals_796, (4, ), (1, ))
    assert_size_stride(primals_797, (4, ), (1, ))
    assert_size_stride(primals_798, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_799, (2, ), (1, ))
    assert_size_stride(primals_800, (2, ), (1, ))
    assert_size_stride(primals_801, (2, ), (1, ))
    assert_size_stride(primals_802, (2, ), (1, ))
    assert_size_stride(primals_803, (2, ), (1, ))
    assert_size_stride(primals_804, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_805, (2, ), (1, ))
    assert_size_stride(primals_806, (2, ), (1, ))
    assert_size_stride(primals_807, (2, ), (1, ))
    assert_size_stride(primals_808, (2, ), (1, ))
    assert_size_stride(primals_809, (2, ), (1, ))
    assert_size_stride(primals_810, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_811, (4, ), (1, ))
    assert_size_stride(primals_812, (4, ), (1, ))
    assert_size_stride(primals_813, (4, ), (1, ))
    assert_size_stride(primals_814, (4, ), (1, ))
    assert_size_stride(primals_815, (4, ), (1, ))
    assert_size_stride(primals_816, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_817, (2, ), (1, ))
    assert_size_stride(primals_818, (2, ), (1, ))
    assert_size_stride(primals_819, (2, ), (1, ))
    assert_size_stride(primals_820, (2, ), (1, ))
    assert_size_stride(primals_821, (2, ), (1, ))
    assert_size_stride(primals_822, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_823, (2, ), (1, ))
    assert_size_stride(primals_824, (2, ), (1, ))
    assert_size_stride(primals_825, (2, ), (1, ))
    assert_size_stride(primals_826, (2, ), (1, ))
    assert_size_stride(primals_827, (2, ), (1, ))
    assert_size_stride(primals_828, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_829, (4, ), (1, ))
    assert_size_stride(primals_830, (4, ), (1, ))
    assert_size_stride(primals_831, (4, ), (1, ))
    assert_size_stride(primals_832, (4, ), (1, ))
    assert_size_stride(primals_833, (4, ), (1, ))
    assert_size_stride(primals_834, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_835, (2, ), (1, ))
    assert_size_stride(primals_836, (2, ), (1, ))
    assert_size_stride(primals_837, (2, ), (1, ))
    assert_size_stride(primals_838, (2, ), (1, ))
    assert_size_stride(primals_839, (2, ), (1, ))
    assert_size_stride(primals_840, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_841, (2, ), (1, ))
    assert_size_stride(primals_842, (2, ), (1, ))
    assert_size_stride(primals_843, (2, ), (1, ))
    assert_size_stride(primals_844, (2, ), (1, ))
    assert_size_stride(primals_845, (2, ), (1, ))
    assert_size_stride(primals_846, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_847, (4, ), (1, ))
    assert_size_stride(primals_848, (4, ), (1, ))
    assert_size_stride(primals_849, (4, ), (1, ))
    assert_size_stride(primals_850, (4, ), (1, ))
    assert_size_stride(primals_851, (4, ), (1, ))
    assert_size_stride(primals_852, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_853, (2, ), (1, ))
    assert_size_stride(primals_854, (2, ), (1, ))
    assert_size_stride(primals_855, (2, ), (1, ))
    assert_size_stride(primals_856, (2, ), (1, ))
    assert_size_stride(primals_857, (2, ), (1, ))
    assert_size_stride(primals_858, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_859, (2, ), (1, ))
    assert_size_stride(primals_860, (2, ), (1, ))
    assert_size_stride(primals_861, (2, ), (1, ))
    assert_size_stride(primals_862, (2, ), (1, ))
    assert_size_stride(primals_863, (2, ), (1, ))
    assert_size_stride(primals_864, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_865, (4, ), (1, ))
    assert_size_stride(primals_866, (4, ), (1, ))
    assert_size_stride(primals_867, (4, ), (1, ))
    assert_size_stride(primals_868, (4, ), (1, ))
    assert_size_stride(primals_869, (4, ), (1, ))
    assert_size_stride(primals_870, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_871, (2, ), (1, ))
    assert_size_stride(primals_872, (2, ), (1, ))
    assert_size_stride(primals_873, (2, ), (1, ))
    assert_size_stride(primals_874, (2, ), (1, ))
    assert_size_stride(primals_875, (2, ), (1, ))
    assert_size_stride(primals_876, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_877, (2, ), (1, ))
    assert_size_stride(primals_878, (2, ), (1, ))
    assert_size_stride(primals_879, (2, ), (1, ))
    assert_size_stride(primals_880, (2, ), (1, ))
    assert_size_stride(primals_881, (2, ), (1, ))
    assert_size_stride(primals_882, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_883, (4, ), (1, ))
    assert_size_stride(primals_884, (4, ), (1, ))
    assert_size_stride(primals_885, (4, ), (1, ))
    assert_size_stride(primals_886, (4, ), (1, ))
    assert_size_stride(primals_887, (4, ), (1, ))
    assert_size_stride(primals_888, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_889, (2, ), (1, ))
    assert_size_stride(primals_890, (2, ), (1, ))
    assert_size_stride(primals_891, (2, ), (1, ))
    assert_size_stride(primals_892, (2, ), (1, ))
    assert_size_stride(primals_893, (2, ), (1, ))
    assert_size_stride(primals_894, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_895, (2, ), (1, ))
    assert_size_stride(primals_896, (2, ), (1, ))
    assert_size_stride(primals_897, (2, ), (1, ))
    assert_size_stride(primals_898, (2, ), (1, ))
    assert_size_stride(primals_899, (2, ), (1, ))
    assert_size_stride(primals_900, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_901, (4, ), (1, ))
    assert_size_stride(primals_902, (4, ), (1, ))
    assert_size_stride(primals_903, (4, ), (1, ))
    assert_size_stride(primals_904, (4, ), (1, ))
    assert_size_stride(primals_905, (4, ), (1, ))
    assert_size_stride(primals_906, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_907, (2, ), (1, ))
    assert_size_stride(primals_908, (2, ), (1, ))
    assert_size_stride(primals_909, (2, ), (1, ))
    assert_size_stride(primals_910, (2, ), (1, ))
    assert_size_stride(primals_911, (2, ), (1, ))
    assert_size_stride(primals_912, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_913, (2, ), (1, ))
    assert_size_stride(primals_914, (2, ), (1, ))
    assert_size_stride(primals_915, (2, ), (1, ))
    assert_size_stride(primals_916, (2, ), (1, ))
    assert_size_stride(primals_917, (2, ), (1, ))
    assert_size_stride(primals_918, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_919, (4, ), (1, ))
    assert_size_stride(primals_920, (4, ), (1, ))
    assert_size_stride(primals_921, (4, ), (1, ))
    assert_size_stride(primals_922, (4, ), (1, ))
    assert_size_stride(primals_923, (4, ), (1, ))
    assert_size_stride(primals_924, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_925, (2, ), (1, ))
    assert_size_stride(primals_926, (2, ), (1, ))
    assert_size_stride(primals_927, (2, ), (1, ))
    assert_size_stride(primals_928, (2, ), (1, ))
    assert_size_stride(primals_929, (2, ), (1, ))
    assert_size_stride(primals_930, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_931, (2, ), (1, ))
    assert_size_stride(primals_932, (2, ), (1, ))
    assert_size_stride(primals_933, (2, ), (1, ))
    assert_size_stride(primals_934, (2, ), (1, ))
    assert_size_stride(primals_935, (2, ), (1, ))
    assert_size_stride(primals_936, (4, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_937, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(primals_1, primals_2, primals_3, primals_4, primals_5, buf0, 65536, grid=grid(65536), stream=stream0)
        del primals_4
        del primals_5
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 2, 64, 64), (8192, 4096, 64, 1))
        buf2 = buf1; del buf1  # reuse
        buf3 = empty_strided_cuda((4, 2, 64, 64), (8192, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4, input_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf2, primals_7, primals_8, primals_9, primals_10, primals_11, buf3, 32768, grid=grid(32768), stream=stream0)
        del primals_11
        del primals_7
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 2, 64, 64), (8192, 4096, 64, 1))
        buf5 = buf4; del buf4  # reuse
        buf6 = empty_strided_cuda((4, 2, 64, 64), (8192, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7, input_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf5, primals_13, primals_14, primals_15, primals_16, primals_17, buf6, 32768, grid=grid(32768), stream=stream0)
        del primals_13
        del primals_17
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf8 = buf7; del buf7  # reuse
        buf9 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, up1, input_10, input_11], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_2.run(buf8, primals_19, primals_1, primals_20, primals_21, primals_22, primals_23, buf9, 65536, grid=grid(65536), stream=stream0)
        del primals_19
        del primals_23
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_24, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 2, 64, 64), (8192, 4096, 64, 1))
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((4, 2, 64, 64), (8192, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_12, input_13, input_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf11, primals_25, primals_26, primals_27, primals_28, primals_29, buf12, 32768, grid=grid(32768), stream=stream0)
        del primals_25
        del primals_29
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 2, 64, 64), (8192, 4096, 64, 1))
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided_cuda((4, 2, 64, 64), (8192, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15, input_16, input_17], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf14, primals_31, primals_32, primals_33, primals_34, primals_35, buf15, 32768, grid=grid(32768), stream=stream0)
        del primals_31
        del primals_35
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf17 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [up1, input_18, up1_1, input_19, input_20], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf16, primals_37, buf8, primals_1, primals_38, primals_39, primals_40, primals_41, buf17, 65536, grid=grid(65536), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 2, 64, 64), (8192, 4096, 64, 1))
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided_cuda((4, 2, 64, 64), (8192, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_21, input_22, input_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf19, primals_43, primals_44, primals_45, primals_46, primals_47, buf20, 32768, grid=grid(32768), stream=stream0)
        del primals_43
        del primals_47
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 2, 64, 64), (8192, 4096, 64, 1))
        buf22 = buf21; del buf21  # reuse
        buf23 = empty_strided_cuda((4, 2, 64, 64), (8192, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf22, primals_49, primals_50, primals_51, primals_52, primals_53, buf23, 32768, grid=grid(32768), stream=stream0)
        del primals_49
        del primals_53
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf25 = buf24; del buf24  # reuse
        buf26 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [up1, input_18, up1_1, input_27, up1_2, input_28, input_29], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4.run(buf25, primals_55, buf16, primals_37, buf8, primals_1, primals_56, primals_57, primals_58, primals_59, buf26, 65536, grid=grid(65536), stream=stream0)
        del primals_55
        del primals_59
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 2, 64, 64), (8192, 4096, 64, 1))
        buf28 = buf27; del buf27  # reuse
        buf29 = empty_strided_cuda((4, 2, 64, 64), (8192, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_30, input_31, input_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf28, primals_61, primals_62, primals_63, primals_64, primals_65, buf29, 32768, grid=grid(32768), stream=stream0)
        del primals_61
        del primals_65
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 2, 64, 64), (8192, 4096, 64, 1))
        buf31 = buf30; del buf30  # reuse
        buf32 = empty_strided_cuda((4, 2, 64, 64), (8192, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_33, input_34, input_35], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf31, primals_67, primals_68, primals_69, primals_70, primals_71, buf32, 32768, grid=grid(32768), stream=stream0)
        del primals_67
        del primals_71
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf34 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [low1, input_37, input_38], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_5.run(primals_1, primals_74, primals_75, primals_76, primals_77, buf34, 16384, grid=grid(16384), stream=stream0)
        del primals_76
        del primals_77
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_39, input_40, input_41], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf36, primals_79, primals_80, primals_81, primals_82, primals_83, buf37, 8192, grid=grid(8192), stream=stream0)
        del primals_79
        del primals_83
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf39 = buf38; del buf38  # reuse
        buf40 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_42, input_43, input_44], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf39, primals_85, primals_86, primals_87, primals_88, primals_89, buf40, 8192, grid=grid(8192), stream=stream0)
        del primals_85
        del primals_89
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf42 = buf41; del buf41  # reuse
        buf43 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [low1, input_45, low1_1, input_46, input_47], Original ATen: [aten.max_pool2d_with_indices, aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_7.run(buf42, primals_91, primals_1, primals_92, primals_93, primals_94, primals_95, buf43, 16384, grid=grid(16384), stream=stream0)
        del primals_91
        del primals_95
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf45 = buf44; del buf44  # reuse
        buf46 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_48, input_49, input_50], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf45, primals_97, primals_98, primals_99, primals_100, primals_101, buf46, 8192, grid=grid(8192), stream=stream0)
        del primals_101
        del primals_97
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf48 = buf47; del buf47  # reuse
        buf49 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_51, input_52, input_53], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf48, primals_103, primals_104, primals_105, primals_106, primals_107, buf49, 8192, grid=grid(8192), stream=stream0)
        del primals_103
        del primals_107
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf51 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_54, low1_2, input_55, input_56], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8.run(buf50, primals_109, buf42, primals_110, primals_111, primals_112, primals_113, buf51, 16384, grid=grid(16384), stream=stream0)
        del primals_113
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf53 = buf52; del buf52  # reuse
        buf54 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_57, input_58, input_59], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf53, primals_115, primals_116, primals_117, primals_118, primals_119, buf54, 8192, grid=grid(8192), stream=stream0)
        del primals_115
        del primals_119
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_120, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf56 = buf55; del buf55  # reuse
        buf57 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_60, input_61, input_62], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf56, primals_121, primals_122, primals_123, primals_124, primals_125, buf57, 8192, grid=grid(8192), stream=stream0)
        del primals_121
        del primals_125
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf59 = buf58; del buf58  # reuse
        buf60 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        buf491 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_54, low1_2, input_63, low1_3, input_64, input_65], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_9.run(buf59, primals_127, buf50, primals_109, buf42, primals_128, primals_129, primals_130, primals_131, buf60, buf491, 16384, grid=grid(16384), stream=stream0)
        del primals_127
        del primals_128
        del primals_131
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf62 = buf61; del buf61  # reuse
        buf63 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_66, input_67, input_68], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf62, primals_133, primals_134, primals_135, primals_136, primals_137, buf63, 8192, grid=grid(8192), stream=stream0)
        del primals_133
        del primals_137
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_138, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf65 = buf64; del buf64  # reuse
        buf66 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_69, input_70, input_71], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf65, primals_139, primals_140, primals_141, primals_142, primals_143, buf66, 8192, grid=grid(8192), stream=stream0)
        del primals_139
        del primals_143
        # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf68 = buf67; del buf67  # reuse
        buf69 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_72, low1_4, input_73, input_74], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10.run(buf68, primals_145, buf59, primals_146, primals_147, primals_148, primals_149, buf69, 16384, grid=grid(16384), stream=stream0)
        del primals_145
        del primals_149
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf71 = buf70; del buf70  # reuse
        buf72 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_75, input_76, input_77], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf71, primals_151, primals_152, primals_153, primals_154, primals_155, buf72, 8192, grid=grid(8192), stream=stream0)
        del primals_151
        del primals_155
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf74 = buf73; del buf73  # reuse
        buf75 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_78, input_79, input_80], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf74, primals_157, primals_158, primals_159, primals_160, primals_161, buf75, 8192, grid=grid(8192), stream=stream0)
        del primals_157
        del primals_161
        # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf77 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_81, up1_4, input_82, input_83], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8.run(buf76, primals_163, buf68, primals_164, primals_165, primals_166, primals_167, buf77, 16384, grid=grid(16384), stream=stream0)
        del primals_167
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf79 = buf78; del buf78  # reuse
        buf80 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_84, input_85, input_86], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf79, primals_169, primals_170, primals_171, primals_172, primals_173, buf80, 8192, grid=grid(8192), stream=stream0)
        del primals_169
        del primals_173
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf82 = buf81; del buf81  # reuse
        buf83 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_87, input_88, input_89], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf82, primals_175, primals_176, primals_177, primals_178, primals_179, buf83, 8192, grid=grid(8192), stream=stream0)
        del primals_175
        del primals_179
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf85 = buf84; del buf84  # reuse
        buf86 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_81, up1_4, input_90, up1_5, input_91, input_92], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11.run(buf85, primals_181, buf76, primals_163, buf68, primals_182, primals_183, primals_184, primals_185, buf86, 16384, grid=grid(16384), stream=stream0)
        del primals_181
        del primals_185
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf88 = buf87; del buf87  # reuse
        buf89 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_93, input_94, input_95], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf88, primals_187, primals_188, primals_189, primals_190, primals_191, buf89, 8192, grid=grid(8192), stream=stream0)
        del primals_187
        del primals_191
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf91 = buf90; del buf90  # reuse
        buf92 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_96, input_97, input_98], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf91, primals_193, primals_194, primals_195, primals_196, primals_197, buf92, 8192, grid=grid(8192), stream=stream0)
        del primals_193
        del primals_197
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf94 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_99, up1_6, input_100, input_101], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8.run(buf93, primals_199, buf85, primals_200, primals_201, primals_202, primals_203, buf94, 16384, grid=grid(16384), stream=stream0)
        del primals_203
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf96 = buf95; del buf95  # reuse
        buf97 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_102, input_103, input_104], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf96, primals_205, primals_206, primals_207, primals_208, primals_209, buf97, 8192, grid=grid(8192), stream=stream0)
        del primals_205
        del primals_209
        # Topologically Sorted Source Nodes: [input_105], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_210, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf99 = buf98; del buf98  # reuse
        buf100 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_105, input_106, input_107], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf99, primals_211, primals_212, primals_213, primals_214, primals_215, buf100, 8192, grid=grid(8192), stream=stream0)
        del primals_211
        del primals_215
        # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf102 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.int8)
        buf103 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [low1_5, input_109, input_110], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12.run(buf68, primals_218, primals_219, primals_220, primals_221, buf102, buf103, 4096, grid=grid(4096), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [input_111], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 2, 16, 16), (512, 256, 16, 1))
        buf105 = buf104; del buf104  # reuse
        buf106 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_111, input_112, input_113], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf105, primals_223, primals_224, primals_225, primals_226, primals_227, buf106, 2048, grid=grid(2048), stream=stream0)
        del primals_223
        del primals_227
        # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_228, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 2, 16, 16), (512, 256, 16, 1))
        buf108 = buf107; del buf107  # reuse
        buf109 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_114, input_115, input_116], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf108, primals_229, primals_230, primals_231, primals_232, primals_233, buf109, 2048, grid=grid(2048), stream=stream0)
        del primals_229
        del primals_233
        # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf111 = buf110; del buf110  # reuse
        buf112 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [low1_5, input_117, low1_6, input_118, input_119], Original ATen: [aten.max_pool2d_with_indices, aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_14.run(buf111, primals_235, buf68, primals_236, primals_237, primals_238, primals_239, buf112, 4096, grid=grid(4096), stream=stream0)
        del primals_235
        del primals_239
        # Topologically Sorted Source Nodes: [input_120], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 2, 16, 16), (512, 256, 16, 1))
        buf114 = buf113; del buf113  # reuse
        buf115 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_120, input_121, input_122], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf114, primals_241, primals_242, primals_243, primals_244, primals_245, buf115, 2048, grid=grid(2048), stream=stream0)
        del primals_241
        del primals_245
        # Topologically Sorted Source Nodes: [input_123], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_246, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 2, 16, 16), (512, 256, 16, 1))
        buf117 = buf116; del buf116  # reuse
        buf118 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_123, input_124, input_125], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf117, primals_247, primals_248, primals_249, primals_250, primals_251, buf118, 2048, grid=grid(2048), stream=stream0)
        del primals_247
        del primals_251
        # Topologically Sorted Source Nodes: [input_126], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf120 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_126, low1_7, input_127, input_128], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.run(buf119, primals_253, buf111, primals_254, primals_255, primals_256, primals_257, buf120, 4096, grid=grid(4096), stream=stream0)
        del primals_257
        # Topologically Sorted Source Nodes: [input_129], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 2, 16, 16), (512, 256, 16, 1))
        buf122 = buf121; del buf121  # reuse
        buf123 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_129, input_130, input_131], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf122, primals_259, primals_260, primals_261, primals_262, primals_263, buf123, 2048, grid=grid(2048), stream=stream0)
        del primals_259
        del primals_263
        # Topologically Sorted Source Nodes: [input_132], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_264, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 2, 16, 16), (512, 256, 16, 1))
        buf125 = buf124; del buf124  # reuse
        buf126 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_132, input_133, input_134], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf125, primals_265, primals_266, primals_267, primals_268, primals_269, buf126, 2048, grid=grid(2048), stream=stream0)
        del primals_265
        del primals_269
        # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf128 = buf127; del buf127  # reuse
        buf129 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf484 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_126, low1_7, input_135, low1_8, input_136, input_137], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_16.run(buf128, primals_271, buf119, primals_253, buf111, primals_272, primals_273, primals_274, primals_275, buf129, buf484, 4096, grid=grid(4096), stream=stream0)
        del primals_271
        del primals_272
        del primals_275
        # Topologically Sorted Source Nodes: [input_138], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_276, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 2, 16, 16), (512, 256, 16, 1))
        buf131 = buf130; del buf130  # reuse
        buf132 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_138, input_139, input_140], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf131, primals_277, primals_278, primals_279, primals_280, primals_281, buf132, 2048, grid=grid(2048), stream=stream0)
        del primals_277
        del primals_281
        # Topologically Sorted Source Nodes: [input_141], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 2, 16, 16), (512, 256, 16, 1))
        buf134 = buf133; del buf133  # reuse
        buf135 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_141, input_142, input_143], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf134, primals_283, primals_284, primals_285, primals_286, primals_287, buf135, 2048, grid=grid(2048), stream=stream0)
        del primals_283
        del primals_287
        # Topologically Sorted Source Nodes: [input_144], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_288, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf137 = buf136; del buf136  # reuse
        buf138 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_144, low1_9, input_145, input_146], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17.run(buf137, primals_289, buf128, primals_290, primals_291, primals_292, primals_293, buf138, 4096, grid=grid(4096), stream=stream0)
        del primals_289
        del primals_293
        # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_294, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 2, 16, 16), (512, 256, 16, 1))
        buf140 = buf139; del buf139  # reuse
        buf141 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_147, input_148, input_149], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf140, primals_295, primals_296, primals_297, primals_298, primals_299, buf141, 2048, grid=grid(2048), stream=stream0)
        del primals_295
        del primals_299
        # Topologically Sorted Source Nodes: [input_150], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_300, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 2, 16, 16), (512, 256, 16, 1))
        buf143 = buf142; del buf142  # reuse
        buf144 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_150, input_151, input_152], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf143, primals_301, primals_302, primals_303, primals_304, primals_305, buf144, 2048, grid=grid(2048), stream=stream0)
        del primals_301
        del primals_305
        # Topologically Sorted Source Nodes: [input_153], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_306, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf146 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [input_153, up1_8, input_154, input_155], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.run(buf145, primals_307, buf137, primals_308, primals_309, primals_310, primals_311, buf146, 4096, grid=grid(4096), stream=stream0)
        del primals_311
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_312, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 2, 16, 16), (512, 256, 16, 1))
        buf148 = buf147; del buf147  # reuse
        buf149 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_156, input_157, input_158], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf148, primals_313, primals_314, primals_315, primals_316, primals_317, buf149, 2048, grid=grid(2048), stream=stream0)
        del primals_313
        del primals_317
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_318, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 2, 16, 16), (512, 256, 16, 1))
        buf151 = buf150; del buf150  # reuse
        buf152 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_159, input_160, input_161], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf151, primals_319, primals_320, primals_321, primals_322, primals_323, buf152, 2048, grid=grid(2048), stream=stream0)
        del primals_319
        del primals_323
        # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_324, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf154 = buf153; del buf153  # reuse
        buf155 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_153, up1_8, input_162, up1_9, input_163, input_164], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18.run(buf154, primals_325, buf145, primals_307, buf137, primals_326, primals_327, primals_328, primals_329, buf155, 4096, grid=grid(4096), stream=stream0)
        del primals_325
        del primals_329
        # Topologically Sorted Source Nodes: [input_165], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_330, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 2, 16, 16), (512, 256, 16, 1))
        buf157 = buf156; del buf156  # reuse
        buf158 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_165, input_166, input_167], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf157, primals_331, primals_332, primals_333, primals_334, primals_335, buf158, 2048, grid=grid(2048), stream=stream0)
        del primals_331
        del primals_335
        # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_336, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 2, 16, 16), (512, 256, 16, 1))
        buf160 = buf159; del buf159  # reuse
        buf161 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_168, input_169, input_170], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf160, primals_337, primals_338, primals_339, primals_340, primals_341, buf161, 2048, grid=grid(2048), stream=stream0)
        del primals_337
        del primals_341
        # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_342, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf163 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_171, up1_10, input_172, input_173], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.run(buf162, primals_343, buf154, primals_344, primals_345, primals_346, primals_347, buf163, 4096, grid=grid(4096), stream=stream0)
        del primals_347
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_348, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 2, 16, 16), (512, 256, 16, 1))
        buf165 = buf164; del buf164  # reuse
        buf166 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_174, input_175, input_176], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf165, primals_349, primals_350, primals_351, primals_352, primals_353, buf166, 2048, grid=grid(2048), stream=stream0)
        del primals_349
        del primals_353
        # Topologically Sorted Source Nodes: [input_177], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_354, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 2, 16, 16), (512, 256, 16, 1))
        buf168 = buf167; del buf167  # reuse
        buf169 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_177, input_178, input_179], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf168, primals_355, primals_356, primals_357, primals_358, primals_359, buf169, 2048, grid=grid(2048), stream=stream0)
        del primals_355
        del primals_359
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_360, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf171 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.int8)
        buf172 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [low1_10, input_181, input_182], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_19.run(buf137, primals_362, primals_363, primals_364, primals_365, buf171, buf172, 1024, grid=grid(1024), stream=stream0)
        del primals_365
        # Topologically Sorted Source Nodes: [input_183], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_366, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 2, 8, 8), (128, 64, 8, 1))
        buf174 = buf173; del buf173  # reuse
        buf175 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_183, input_184, input_185], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf174, primals_367, primals_368, primals_369, primals_370, primals_371, buf175, 512, grid=grid(512), stream=stream0)
        del primals_367
        del primals_371
        # Topologically Sorted Source Nodes: [input_186], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 2, 8, 8), (128, 64, 8, 1))
        buf177 = buf176; del buf176  # reuse
        buf178 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_186, input_187, input_188], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf177, primals_373, primals_374, primals_375, primals_376, primals_377, buf178, 512, grid=grid(512), stream=stream0)
        del primals_373
        del primals_377
        # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_378, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 4, 8, 8), (256, 64, 8, 1))
        buf180 = buf179; del buf179  # reuse
        buf181 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [low1_10, input_189, low1_11, input_190, input_191], Original ATen: [aten.max_pool2d_with_indices, aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_21.run(buf180, primals_379, buf137, primals_380, primals_381, primals_382, primals_383, buf181, 1024, grid=grid(1024), stream=stream0)
        del primals_379
        del primals_383
        # Topologically Sorted Source Nodes: [input_192], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_384, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 2, 8, 8), (128, 64, 8, 1))
        buf183 = buf182; del buf182  # reuse
        buf184 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_192, input_193, input_194], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf183, primals_385, primals_386, primals_387, primals_388, primals_389, buf184, 512, grid=grid(512), stream=stream0)
        del primals_385
        del primals_389
        # Topologically Sorted Source Nodes: [input_195], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_390, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 2, 8, 8), (128, 64, 8, 1))
        buf186 = buf185; del buf185  # reuse
        buf187 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_195, input_196, input_197], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf186, primals_391, primals_392, primals_393, primals_394, primals_395, buf187, 512, grid=grid(512), stream=stream0)
        del primals_391
        del primals_395
        # Topologically Sorted Source Nodes: [input_198], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_396, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 4, 8, 8), (256, 64, 8, 1))
        buf189 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_198, low1_12, input_199, input_200], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22.run(buf188, primals_397, buf180, primals_398, primals_399, primals_400, primals_401, buf189, 1024, grid=grid(1024), stream=stream0)
        del primals_401
        # Topologically Sorted Source Nodes: [input_201], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 2, 8, 8), (128, 64, 8, 1))
        buf191 = buf190; del buf190  # reuse
        buf192 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_201, input_202, input_203], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf191, primals_403, primals_404, primals_405, primals_406, primals_407, buf192, 512, grid=grid(512), stream=stream0)
        del primals_403
        del primals_407
        # Topologically Sorted Source Nodes: [input_204], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_408, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 2, 8, 8), (128, 64, 8, 1))
        buf194 = buf193; del buf193  # reuse
        buf195 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_204, input_205, input_206], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf194, primals_409, primals_410, primals_411, primals_412, primals_413, buf195, 512, grid=grid(512), stream=stream0)
        del primals_409
        del primals_413
        # Topologically Sorted Source Nodes: [input_207], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_414, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 4, 8, 8), (256, 64, 8, 1))
        buf197 = buf196; del buf196  # reuse
        buf198 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        buf477 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_198, low1_12, input_207, low1_13, input_208, input_209], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_23.run(buf197, primals_415, buf188, primals_397, buf180, primals_416, primals_417, primals_418, primals_419, buf198, buf477, 1024, grid=grid(1024), stream=stream0)
        del primals_415
        del primals_416
        del primals_419
        # Topologically Sorted Source Nodes: [input_210], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_420, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 2, 8, 8), (128, 64, 8, 1))
        buf200 = buf199; del buf199  # reuse
        buf201 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_210, input_211, input_212], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf200, primals_421, primals_422, primals_423, primals_424, primals_425, buf201, 512, grid=grid(512), stream=stream0)
        del primals_421
        del primals_425
        # Topologically Sorted Source Nodes: [input_213], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_426, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 2, 8, 8), (128, 64, 8, 1))
        buf203 = buf202; del buf202  # reuse
        buf204 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_213, input_214, input_215], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf203, primals_427, primals_428, primals_429, primals_430, primals_431, buf204, 512, grid=grid(512), stream=stream0)
        del primals_427
        del primals_431
        # Topologically Sorted Source Nodes: [input_216], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_432, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 4, 8, 8), (256, 64, 8, 1))
        buf206 = buf205; del buf205  # reuse
        buf207 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_216, low1_14, input_217, input_218], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(buf206, primals_433, buf197, primals_434, primals_435, primals_436, primals_437, buf207, 1024, grid=grid(1024), stream=stream0)
        del primals_433
        del primals_437
        # Topologically Sorted Source Nodes: [input_219], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_438, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 2, 8, 8), (128, 64, 8, 1))
        buf209 = buf208; del buf208  # reuse
        buf210 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_219, input_220, input_221], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf209, primals_439, primals_440, primals_441, primals_442, primals_443, buf210, 512, grid=grid(512), stream=stream0)
        del primals_439
        del primals_443
        # Topologically Sorted Source Nodes: [input_222], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_444, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 2, 8, 8), (128, 64, 8, 1))
        buf212 = buf211; del buf211  # reuse
        buf213 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_222, input_223, input_224], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf212, primals_445, primals_446, primals_447, primals_448, primals_449, buf213, 512, grid=grid(512), stream=stream0)
        del primals_445
        del primals_449
        # Topologically Sorted Source Nodes: [input_225], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_450, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 4, 8, 8), (256, 64, 8, 1))
        buf215 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [input_225, up1_12, input_226, input_227], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22.run(buf214, primals_451, buf206, primals_452, primals_453, primals_454, primals_455, buf215, 1024, grid=grid(1024), stream=stream0)
        del primals_455
        # Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_456, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 2, 8, 8), (128, 64, 8, 1))
        buf217 = buf216; del buf216  # reuse
        buf218 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_228, input_229, input_230], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf217, primals_457, primals_458, primals_459, primals_460, primals_461, buf218, 512, grid=grid(512), stream=stream0)
        del primals_457
        del primals_461
        # Topologically Sorted Source Nodes: [input_231], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_462, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 2, 8, 8), (128, 64, 8, 1))
        buf220 = buf219; del buf219  # reuse
        buf221 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_231, input_232, input_233], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf220, primals_463, primals_464, primals_465, primals_466, primals_467, buf221, 512, grid=grid(512), stream=stream0)
        del primals_463
        del primals_467
        # Topologically Sorted Source Nodes: [input_234], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_468, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 4, 8, 8), (256, 64, 8, 1))
        buf223 = buf222; del buf222  # reuse
        buf224 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_225, up1_12, input_234, up1_13, input_235, input_236], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25.run(buf223, primals_469, buf214, primals_451, buf206, primals_470, primals_471, primals_472, primals_473, buf224, 1024, grid=grid(1024), stream=stream0)
        del primals_469
        del primals_473
        # Topologically Sorted Source Nodes: [input_237], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_474, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 2, 8, 8), (128, 64, 8, 1))
        buf226 = buf225; del buf225  # reuse
        buf227 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_237, input_238, input_239], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf226, primals_475, primals_476, primals_477, primals_478, primals_479, buf227, 512, grid=grid(512), stream=stream0)
        del primals_475
        del primals_479
        # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_480, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 2, 8, 8), (128, 64, 8, 1))
        buf229 = buf228; del buf228  # reuse
        buf230 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_240, input_241, input_242], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf229, primals_481, primals_482, primals_483, primals_484, primals_485, buf230, 512, grid=grid(512), stream=stream0)
        del primals_481
        del primals_485
        # Topologically Sorted Source Nodes: [input_243], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, primals_486, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 4, 8, 8), (256, 64, 8, 1))
        buf232 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_243, up1_14, input_244, input_245], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22.run(buf231, primals_487, buf223, primals_488, primals_489, primals_490, primals_491, buf232, 1024, grid=grid(1024), stream=stream0)
        del primals_491
        # Topologically Sorted Source Nodes: [input_246], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 2, 8, 8), (128, 64, 8, 1))
        buf234 = buf233; del buf233  # reuse
        buf235 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_246, input_247, input_248], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf234, primals_493, primals_494, primals_495, primals_496, primals_497, buf235, 512, grid=grid(512), stream=stream0)
        del primals_493
        del primals_497
        # Topologically Sorted Source Nodes: [input_249], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_498, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 2, 8, 8), (128, 64, 8, 1))
        buf237 = buf236; del buf236  # reuse
        buf238 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_249, input_250, input_251], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf237, primals_499, primals_500, primals_501, primals_502, primals_503, buf238, 512, grid=grid(512), stream=stream0)
        del primals_499
        del primals_503
        # Topologically Sorted Source Nodes: [input_252], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_504, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 4, 8, 8), (256, 64, 8, 1))
        buf240 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.int8)
        buf241 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [low1_15, input_253, input_254], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_26.run(buf206, primals_506, primals_507, primals_508, primals_509, buf240, buf241, 256, grid=grid(256), stream=stream0)
        del primals_509
        # Topologically Sorted Source Nodes: [input_255], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_510, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 2, 4, 4), (32, 16, 4, 1))
        buf243 = buf242; del buf242  # reuse
        buf244 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_255, input_256, input_257], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf243, primals_511, primals_512, primals_513, primals_514, primals_515, buf244, 128, grid=grid(128), stream=stream0)
        del primals_511
        del primals_515
        # Topologically Sorted Source Nodes: [input_258], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_516, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 2, 4, 4), (32, 16, 4, 1))
        buf246 = buf245; del buf245  # reuse
        buf247 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_258, input_259, input_260], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf246, primals_517, primals_518, primals_519, primals_520, primals_521, buf247, 128, grid=grid(128), stream=stream0)
        del primals_517
        del primals_521
        # Topologically Sorted Source Nodes: [input_261], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_522, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 4, 4, 4), (64, 16, 4, 1))
        buf249 = buf248; del buf248  # reuse
        buf250 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [low1_15, input_261, low1_16, input_262, input_263], Original ATen: [aten.max_pool2d_with_indices, aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_relu_28.run(buf249, primals_523, buf206, primals_524, primals_525, primals_526, primals_527, buf250, 256, grid=grid(256), stream=stream0)
        del primals_523
        del primals_527
        # Topologically Sorted Source Nodes: [input_264], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_528, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 2, 4, 4), (32, 16, 4, 1))
        buf252 = buf251; del buf251  # reuse
        buf253 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_264, input_265, input_266], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf252, primals_529, primals_530, primals_531, primals_532, primals_533, buf253, 128, grid=grid(128), stream=stream0)
        del primals_529
        del primals_533
        # Topologically Sorted Source Nodes: [input_267], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_534, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 2, 4, 4), (32, 16, 4, 1))
        buf255 = buf254; del buf254  # reuse
        buf256 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_267, input_268, input_269], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf255, primals_535, primals_536, primals_537, primals_538, primals_539, buf256, 128, grid=grid(128), stream=stream0)
        del primals_535
        del primals_539
        # Topologically Sorted Source Nodes: [input_270], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_540, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 4, 4, 4), (64, 16, 4, 1))
        buf258 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_270, low1_17, input_271, input_272], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf257, primals_541, buf249, primals_542, primals_543, primals_544, primals_545, buf258, 256, grid=grid(256), stream=stream0)
        del primals_545
        # Topologically Sorted Source Nodes: [input_273], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_546, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (4, 2, 4, 4), (32, 16, 4, 1))
        buf260 = buf259; del buf259  # reuse
        buf261 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_273, input_274, input_275], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf260, primals_547, primals_548, primals_549, primals_550, primals_551, buf261, 128, grid=grid(128), stream=stream0)
        del primals_547
        del primals_551
        # Topologically Sorted Source Nodes: [input_276], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_552, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 2, 4, 4), (32, 16, 4, 1))
        buf263 = buf262; del buf262  # reuse
        buf264 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_276, input_277, input_278], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf263, primals_553, primals_554, primals_555, primals_556, primals_557, buf264, 128, grid=grid(128), stream=stream0)
        del primals_553
        del primals_557
        # Topologically Sorted Source Nodes: [input_279], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_558, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 4, 4, 4), (64, 16, 4, 1))
        buf266 = buf265; del buf265  # reuse
        buf267 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_270, low1_17, input_279, low1_18, input_280, input_281], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30.run(buf266, primals_559, buf257, primals_541, buf249, primals_560, primals_561, primals_562, primals_563, buf267, 256, grid=grid(256), stream=stream0)
        del primals_559
        del primals_563
        # Topologically Sorted Source Nodes: [input_282], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_564, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 2, 4, 4), (32, 16, 4, 1))
        buf269 = buf268; del buf268  # reuse
        buf270 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_282, input_283, input_284], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf269, primals_565, primals_566, primals_567, primals_568, primals_569, buf270, 128, grid=grid(128), stream=stream0)
        del primals_565
        del primals_569
        # Topologically Sorted Source Nodes: [input_285], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_570, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 2, 4, 4), (32, 16, 4, 1))
        buf272 = buf271; del buf271  # reuse
        buf273 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_285, input_286, input_287], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf272, primals_571, primals_572, primals_573, primals_574, primals_575, buf273, 128, grid=grid(128), stream=stream0)
        del primals_571
        del primals_575
        # Topologically Sorted Source Nodes: [input_288], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_576, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 4, 4, 4), (64, 16, 4, 1))
        buf275 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_288, low1_19, input_289, input_290], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf274, primals_577, buf266, primals_578, primals_579, primals_580, primals_581, buf275, 256, grid=grid(256), stream=stream0)
        del primals_581
        # Topologically Sorted Source Nodes: [input_291], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_582, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 2, 4, 4), (32, 16, 4, 1))
        buf277 = buf276; del buf276  # reuse
        buf278 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_291, input_292, input_293], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf277, primals_583, primals_584, primals_585, primals_586, primals_587, buf278, 128, grid=grid(128), stream=stream0)
        del primals_583
        del primals_587
        # Topologically Sorted Source Nodes: [input_294], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, primals_588, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (4, 2, 4, 4), (32, 16, 4, 1))
        buf280 = buf279; del buf279  # reuse
        buf281 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_294, input_295, input_296], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf280, primals_589, primals_590, primals_591, primals_592, primals_593, buf281, 128, grid=grid(128), stream=stream0)
        del primals_589
        del primals_593
        # Topologically Sorted Source Nodes: [input_297], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_594, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 4, 4, 4), (64, 16, 4, 1))
        buf283 = buf282; del buf282  # reuse
        buf284 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_288, low1_19, input_297, low2, input_298, input_299], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30.run(buf283, primals_595, buf274, primals_577, buf266, primals_596, primals_597, primals_598, primals_599, buf284, 256, grid=grid(256), stream=stream0)
        del primals_595
        del primals_599
        # Topologically Sorted Source Nodes: [input_300], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_600, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 2, 4, 4), (32, 16, 4, 1))
        buf286 = buf285; del buf285  # reuse
        buf287 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_300, input_301, input_302], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf286, primals_601, primals_602, primals_603, primals_604, primals_605, buf287, 128, grid=grid(128), stream=stream0)
        del primals_601
        del primals_605
        # Topologically Sorted Source Nodes: [input_303], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_606, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 2, 4, 4), (32, 16, 4, 1))
        buf289 = buf288; del buf288  # reuse
        buf290 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_303, input_304, input_305], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf289, primals_607, primals_608, primals_609, primals_610, primals_611, buf290, 128, grid=grid(128), stream=stream0)
        del primals_607
        del primals_611
        # Topologically Sorted Source Nodes: [input_306], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_612, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 4, 4, 4), (64, 16, 4, 1))
        buf292 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_306, low2_1, input_307, input_308], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf291, primals_613, buf283, primals_614, primals_615, primals_616, primals_617, buf292, 256, grid=grid(256), stream=stream0)
        del primals_617
        # Topologically Sorted Source Nodes: [input_309], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_618, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 2, 4, 4), (32, 16, 4, 1))
        buf294 = buf293; del buf293  # reuse
        buf295 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_309, input_310, input_311], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf294, primals_619, primals_620, primals_621, primals_622, primals_623, buf295, 128, grid=grid(128), stream=stream0)
        del primals_619
        del primals_623
        # Topologically Sorted Source Nodes: [input_312], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_624, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 2, 4, 4), (32, 16, 4, 1))
        buf297 = buf296; del buf296  # reuse
        buf298 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_312, input_313, input_314], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf297, primals_625, primals_626, primals_627, primals_628, primals_629, buf298, 128, grid=grid(128), stream=stream0)
        del primals_625
        del primals_629
        # Topologically Sorted Source Nodes: [input_315], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_630, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 4, 4, 4), (64, 16, 4, 1))
        buf300 = buf299; del buf299  # reuse
        buf301 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_306, low2_1, input_315, low2_2, input_316, input_317], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30.run(buf300, primals_631, buf291, primals_613, buf283, primals_632, primals_633, primals_634, primals_635, buf301, 256, grid=grid(256), stream=stream0)
        del primals_631
        del primals_635
        # Topologically Sorted Source Nodes: [input_318], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_636, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 2, 4, 4), (32, 16, 4, 1))
        buf303 = buf302; del buf302  # reuse
        buf304 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_318, input_319, input_320], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf303, primals_637, primals_638, primals_639, primals_640, primals_641, buf304, 128, grid=grid(128), stream=stream0)
        del primals_637
        del primals_641
        # Topologically Sorted Source Nodes: [input_321], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, primals_642, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (4, 2, 4, 4), (32, 16, 4, 1))
        buf306 = buf305; del buf305  # reuse
        buf307 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_321, input_322, input_323], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf306, primals_643, primals_644, primals_645, primals_646, primals_647, buf307, 128, grid=grid(128), stream=stream0)
        del primals_643
        del primals_647
        # Topologically Sorted Source Nodes: [input_324], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_648, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 4, 4, 4), (64, 16, 4, 1))
        buf309 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_324, low2_3, input_325, input_326], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf308, primals_649, buf300, primals_650, primals_651, primals_652, primals_653, buf309, 256, grid=grid(256), stream=stream0)
        del primals_653
        # Topologically Sorted Source Nodes: [input_327], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_654, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 2, 4, 4), (32, 16, 4, 1))
        buf311 = buf310; del buf310  # reuse
        buf312 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_327, input_328, input_329], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf311, primals_655, primals_656, primals_657, primals_658, primals_659, buf312, 128, grid=grid(128), stream=stream0)
        del primals_655
        del primals_659
        # Topologically Sorted Source Nodes: [input_330], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, primals_660, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 2, 4, 4), (32, 16, 4, 1))
        buf314 = buf313; del buf313  # reuse
        buf315 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_330, input_331, input_332], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf314, primals_661, primals_662, primals_663, primals_664, primals_665, buf315, 128, grid=grid(128), stream=stream0)
        del primals_661
        del primals_665
        # Topologically Sorted Source Nodes: [input_333], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, primals_666, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 4, 4, 4), (64, 16, 4, 1))
        buf317 = buf316; del buf316  # reuse
        buf318 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_324, low2_3, input_333, low3, input_334, input_335], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30.run(buf317, primals_667, buf308, primals_649, buf300, primals_668, primals_669, primals_670, primals_671, buf318, 256, grid=grid(256), stream=stream0)
        del primals_667
        del primals_671
        # Topologically Sorted Source Nodes: [input_336], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_672, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 2, 4, 4), (32, 16, 4, 1))
        buf320 = buf319; del buf319  # reuse
        buf321 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_336, input_337, input_338], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf320, primals_673, primals_674, primals_675, primals_676, primals_677, buf321, 128, grid=grid(128), stream=stream0)
        del primals_673
        del primals_677
        # Topologically Sorted Source Nodes: [input_339], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_678, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 2, 4, 4), (32, 16, 4, 1))
        buf323 = buf322; del buf322  # reuse
        buf324 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_339, input_340, input_341], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf323, primals_679, primals_680, primals_681, primals_682, primals_683, buf324, 128, grid=grid(128), stream=stream0)
        del primals_679
        del primals_683
        # Topologically Sorted Source Nodes: [input_342], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf324, primals_684, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (4, 4, 4, 4), (64, 16, 4, 1))
        buf326 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_342, low3_1, input_343, input_344], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf325, primals_685, buf317, primals_686, primals_687, primals_688, primals_689, buf326, 256, grid=grid(256), stream=stream0)
        del primals_689
        # Topologically Sorted Source Nodes: [input_345], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, primals_690, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (4, 2, 4, 4), (32, 16, 4, 1))
        buf328 = buf327; del buf327  # reuse
        buf329 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_345, input_346, input_347], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf328, primals_691, primals_692, primals_693, primals_694, primals_695, buf329, 128, grid=grid(128), stream=stream0)
        del primals_691
        del primals_695
        # Topologically Sorted Source Nodes: [input_348], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_696, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 2, 4, 4), (32, 16, 4, 1))
        buf331 = buf330; del buf330  # reuse
        buf332 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_348, input_349, input_350], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf331, primals_697, primals_698, primals_699, primals_700, primals_701, buf332, 128, grid=grid(128), stream=stream0)
        del primals_697
        del primals_701
        # Topologically Sorted Source Nodes: [input_351], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_702, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 4, 4, 4), (64, 16, 4, 1))
        buf334 = buf333; del buf333  # reuse
        buf335 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf462 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_342, low3_1, input_351, low3_2, input_352, input_353], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_31.run(buf334, primals_703, buf325, primals_685, buf317, primals_704, primals_705, primals_706, primals_707, buf335, buf462, 256, grid=grid(256), stream=stream0)
        del primals_703
        del primals_704
        del primals_707
        # Topologically Sorted Source Nodes: [input_354], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_708, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 2, 4, 4), (32, 16, 4, 1))
        buf337 = buf336; del buf336  # reuse
        buf338 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_354, input_355, input_356], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf337, primals_709, primals_710, primals_711, primals_712, primals_713, buf338, 128, grid=grid(128), stream=stream0)
        del primals_709
        del primals_713
        # Topologically Sorted Source Nodes: [input_357], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_714, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (4, 2, 4, 4), (32, 16, 4, 1))
        buf340 = buf339; del buf339  # reuse
        buf341 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_357, input_358, input_359], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf340, primals_715, primals_716, primals_717, primals_718, primals_719, buf341, 128, grid=grid(128), stream=stream0)
        del primals_715
        del primals_719
        # Topologically Sorted Source Nodes: [input_360], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_720, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 4, 4, 4), (64, 16, 4, 1))
        buf343 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_32.run(buf343, 8, grid=grid(8), stream=stream0)
        buf344 = buf239; del buf239  # reuse
        buf345 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_243, up1_14, input_252, up1_15, input_360, low3_3, up2, low2_4, input_361, input_362], Original ATen: [aten.convolution, aten.add, aten._unsafe_index, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_33.run(buf344, primals_505, buf231, primals_487, buf223, buf343, buf342, primals_721, buf334, primals_722, primals_723, primals_724, primals_725, buf345, 1024, grid=grid(1024), stream=stream0)
        del buf334
        del primals_505
        del primals_721
        del primals_725
        # Topologically Sorted Source Nodes: [input_363], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_726, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (4, 2, 8, 8), (128, 64, 8, 1))
        buf347 = buf346; del buf346  # reuse
        buf348 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_363, input_364, input_365], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf347, primals_727, primals_728, primals_729, primals_730, primals_731, buf348, 512, grid=grid(512), stream=stream0)
        del primals_727
        del primals_731
        # Topologically Sorted Source Nodes: [input_366], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, primals_732, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 2, 8, 8), (128, 64, 8, 1))
        buf350 = buf349; del buf349  # reuse
        buf351 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_366, input_367, input_368], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf350, primals_733, primals_734, primals_735, primals_736, primals_737, buf351, 512, grid=grid(512), stream=stream0)
        del primals_733
        del primals_737
        # Topologically Sorted Source Nodes: [input_369], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_738, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (4, 4, 8, 8), (256, 64, 8, 1))
        buf353 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_369, low3_4, input_370, input_371], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22.run(buf352, primals_739, buf344, primals_740, primals_741, primals_742, primals_743, buf353, 1024, grid=grid(1024), stream=stream0)
        del primals_743
        # Topologically Sorted Source Nodes: [input_372], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_744, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 2, 8, 8), (128, 64, 8, 1))
        buf355 = buf354; del buf354  # reuse
        buf356 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_372, input_373, input_374], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf355, primals_745, primals_746, primals_747, primals_748, primals_749, buf356, 512, grid=grid(512), stream=stream0)
        del primals_745
        del primals_749
        # Topologically Sorted Source Nodes: [input_375], Original ATen: [aten.convolution]
        buf357 = extern_kernels.convolution(buf356, primals_750, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (4, 2, 8, 8), (128, 64, 8, 1))
        buf358 = buf357; del buf357  # reuse
        buf359 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_375, input_376, input_377], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf358, primals_751, primals_752, primals_753, primals_754, primals_755, buf359, 512, grid=grid(512), stream=stream0)
        del primals_751
        del primals_755
        # Topologically Sorted Source Nodes: [input_378], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, primals_756, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (4, 4, 8, 8), (256, 64, 8, 1))
        buf361 = buf360; del buf360  # reuse
        buf362 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_369, low3_4, input_378, low3_5, input_379, input_380], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25.run(buf361, primals_757, buf352, primals_739, buf344, primals_758, primals_759, primals_760, primals_761, buf362, 1024, grid=grid(1024), stream=stream0)
        del primals_757
        del primals_761
        # Topologically Sorted Source Nodes: [input_381], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, primals_762, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (4, 2, 8, 8), (128, 64, 8, 1))
        buf364 = buf363; del buf363  # reuse
        buf365 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_381, input_382, input_383], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf364, primals_763, primals_764, primals_765, primals_766, primals_767, buf365, 512, grid=grid(512), stream=stream0)
        del primals_763
        del primals_767
        # Topologically Sorted Source Nodes: [input_384], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_768, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 2, 8, 8), (128, 64, 8, 1))
        buf367 = buf366; del buf366  # reuse
        buf368 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_384, input_385, input_386], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf367, primals_769, primals_770, primals_771, primals_772, primals_773, buf368, 512, grid=grid(512), stream=stream0)
        del primals_769
        del primals_773
        # Topologically Sorted Source Nodes: [input_387], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, primals_774, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 4, 8, 8), (256, 64, 8, 1))
        buf370 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_387, low3_6, input_388, input_389], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22.run(buf369, primals_775, buf361, primals_776, primals_777, primals_778, primals_779, buf370, 1024, grid=grid(1024), stream=stream0)
        del primals_779
        # Topologically Sorted Source Nodes: [input_390], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_780, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (4, 2, 8, 8), (128, 64, 8, 1))
        buf372 = buf371; del buf371  # reuse
        buf373 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_390, input_391, input_392], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf372, primals_781, primals_782, primals_783, primals_784, primals_785, buf373, 512, grid=grid(512), stream=stream0)
        del primals_781
        del primals_785
        # Topologically Sorted Source Nodes: [input_393], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, primals_786, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (4, 2, 8, 8), (128, 64, 8, 1))
        buf375 = buf374; del buf374  # reuse
        buf376 = empty_strided_cuda((4, 2, 8, 8), (128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_393, input_394, input_395], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf375, primals_787, primals_788, primals_789, primals_790, primals_791, buf376, 512, grid=grid(512), stream=stream0)
        del primals_787
        del primals_791
        # Topologically Sorted Source Nodes: [input_396], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_792, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 4, 8, 8), (256, 64, 8, 1))
        buf378 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_34.run(buf378, 16, grid=grid(16), stream=stream0)
        buf379 = buf170; del buf170  # reuse
        buf380 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_171, up1_10, input_180, up1_11, input_387, low3_6, input_396, low3_7, up2_1, low2_5, input_397, input_398], Original ATen: [aten.convolution, aten.add, aten._unsafe_index, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_35.run(buf379, primals_361, buf162, primals_343, buf154, buf378, buf377, primals_793, buf369, primals_775, buf361, primals_794, primals_795, primals_796, primals_797, buf380, 4096, grid=grid(4096), stream=stream0)
        del primals_361
        del primals_793
        del primals_797
        # Topologically Sorted Source Nodes: [input_399], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, primals_798, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 2, 16, 16), (512, 256, 16, 1))
        buf382 = buf381; del buf381  # reuse
        buf383 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_399, input_400, input_401], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf382, primals_799, primals_800, primals_801, primals_802, primals_803, buf383, 2048, grid=grid(2048), stream=stream0)
        del primals_799
        del primals_803
        # Topologically Sorted Source Nodes: [input_402], Original ATen: [aten.convolution]
        buf384 = extern_kernels.convolution(buf383, primals_804, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf384, (4, 2, 16, 16), (512, 256, 16, 1))
        buf385 = buf384; del buf384  # reuse
        buf386 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_402, input_403, input_404], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf385, primals_805, primals_806, primals_807, primals_808, primals_809, buf386, 2048, grid=grid(2048), stream=stream0)
        del primals_805
        del primals_809
        # Topologically Sorted Source Nodes: [input_405], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf386, primals_810, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf388 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_405, low3_8, input_406, input_407], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.run(buf387, primals_811, buf379, primals_812, primals_813, primals_814, primals_815, buf388, 4096, grid=grid(4096), stream=stream0)
        del primals_815
        # Topologically Sorted Source Nodes: [input_408], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_816, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (4, 2, 16, 16), (512, 256, 16, 1))
        buf390 = buf389; del buf389  # reuse
        buf391 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_408, input_409, input_410], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf390, primals_817, primals_818, primals_819, primals_820, primals_821, buf391, 2048, grid=grid(2048), stream=stream0)
        del primals_817
        del primals_821
        # Topologically Sorted Source Nodes: [input_411], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_822, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (4, 2, 16, 16), (512, 256, 16, 1))
        buf393 = buf392; del buf392  # reuse
        buf394 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_411, input_412, input_413], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf393, primals_823, primals_824, primals_825, primals_826, primals_827, buf394, 2048, grid=grid(2048), stream=stream0)
        del primals_823
        del primals_827
        # Topologically Sorted Source Nodes: [input_414], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_828, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf396 = buf395; del buf395  # reuse
        buf456 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf457 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf397 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf455 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_405, low3_8, input_414, low3_9, input_415, input_416], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_36.run(buf396, primals_829, buf387, primals_811, buf379, primals_812, primals_794, primals_830, primals_831, primals_832, primals_833, buf456, buf457, buf397, buf455, 4096, grid=grid(4096), stream=stream0)
        del primals_794
        del primals_811
        del primals_812
        del primals_829
        del primals_830
        del primals_833
        # Topologically Sorted Source Nodes: [input_417], Original ATen: [aten.convolution]
        buf398 = extern_kernels.convolution(buf397, primals_834, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (4, 2, 16, 16), (512, 256, 16, 1))
        buf399 = buf398; del buf398  # reuse
        buf400 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_417, input_418, input_419], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf399, primals_835, primals_836, primals_837, primals_838, primals_839, buf400, 2048, grid=grid(2048), stream=stream0)
        del primals_835
        del primals_839
        # Topologically Sorted Source Nodes: [input_420], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_840, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (4, 2, 16, 16), (512, 256, 16, 1))
        buf402 = buf401; del buf401  # reuse
        buf403 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_420, input_421, input_422], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf402, primals_841, primals_842, primals_843, primals_844, primals_845, buf403, 2048, grid=grid(2048), stream=stream0)
        del primals_841
        del primals_845
        # Topologically Sorted Source Nodes: [input_423], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_846, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf405 = buf387; del buf387  # reuse
        buf454 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [input_423, low3_10, input_424, input_425], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_37.run(buf404, primals_847, buf396, primals_848, primals_849, primals_850, primals_851, buf405, buf454, 4096, grid=grid(4096), stream=stream0)
        del primals_848
        del primals_851
        # Topologically Sorted Source Nodes: [input_426], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf405, primals_852, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (4, 2, 16, 16), (512, 256, 16, 1))
        buf407 = buf406; del buf406  # reuse
        buf408 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_426, input_427, input_428], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf407, primals_853, primals_854, primals_855, primals_856, primals_857, buf408, 2048, grid=grid(2048), stream=stream0)
        del primals_853
        del primals_857
        # Topologically Sorted Source Nodes: [input_429], Original ATen: [aten.convolution]
        buf409 = extern_kernels.convolution(buf408, primals_858, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf409, (4, 2, 16, 16), (512, 256, 16, 1))
        buf410 = buf409; del buf409  # reuse
        buf411 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_429, input_430, input_431], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf410, primals_859, primals_860, primals_861, primals_862, primals_863, buf411, 2048, grid=grid(2048), stream=stream0)
        del primals_859
        del primals_863
        # Topologically Sorted Source Nodes: [input_432], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, primals_864, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf413 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_38.run(buf413, 32, grid=grid(32), stream=stream0)
        buf414 = buf101; del buf101  # reuse
        buf415 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        buf453 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_99, up1_6, input_108, up1_7, input_423, low3_10, input_432, low3_11, up2_2, low2_6, input_433, input_434], Original ATen: [aten.convolution, aten.add, aten._unsafe_index, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_native_batch_norm_backward_relu_39.run(buf414, primals_217, buf93, primals_199, buf85, buf413, buf412, primals_865, buf404, primals_847, buf396, primals_866, primals_867, primals_868, primals_869, buf415, buf453, 16384, grid=grid(16384), stream=stream0)
        del buf396
        del buf404
        del primals_217
        del primals_847
        del primals_865
        del primals_866
        del primals_869
        # Topologically Sorted Source Nodes: [input_435], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_870, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf417 = buf416; del buf416  # reuse
        buf418 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_435, input_436, input_437], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf417, primals_871, primals_872, primals_873, primals_874, primals_875, buf418, 8192, grid=grid(8192), stream=stream0)
        del primals_871
        del primals_875
        # Topologically Sorted Source Nodes: [input_438], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_876, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf420 = buf419; del buf419  # reuse
        buf421 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_438, input_439, input_440], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf420, primals_877, primals_878, primals_879, primals_880, primals_881, buf421, 8192, grid=grid(8192), stream=stream0)
        del primals_877
        del primals_881
        # Topologically Sorted Source Nodes: [input_441], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_882, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf423 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        buf452 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_441, low3_12, input_442, input_443], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_40.run(buf422, primals_883, buf414, primals_884, primals_885, primals_886, primals_887, buf423, buf452, 16384, grid=grid(16384), stream=stream0)
        del primals_884
        del primals_887
        # Topologically Sorted Source Nodes: [input_444], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, primals_888, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf425 = buf424; del buf424  # reuse
        buf426 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_444, input_445, input_446], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf425, primals_889, primals_890, primals_891, primals_892, primals_893, buf426, 8192, grid=grid(8192), stream=stream0)
        del primals_889
        del primals_893
        # Topologically Sorted Source Nodes: [input_447], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_894, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf428 = buf427; del buf427  # reuse
        buf429 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_447, input_448, input_449], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf428, primals_895, primals_896, primals_897, primals_898, primals_899, buf429, 8192, grid=grid(8192), stream=stream0)
        del primals_895
        del primals_899
        # Topologically Sorted Source Nodes: [input_450], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, primals_900, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf431 = buf430; del buf430  # reuse
        buf432 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        buf451 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_441, low3_12, input_450, low3_13, input_451, input_452], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_9.run(buf431, primals_901, buf422, primals_883, buf414, primals_902, primals_903, primals_904, primals_905, buf432, buf451, 16384, grid=grid(16384), stream=stream0)
        del primals_883
        del primals_901
        del primals_902
        del primals_905
        # Topologically Sorted Source Nodes: [input_453], Original ATen: [aten.convolution]
        buf433 = extern_kernels.convolution(buf432, primals_906, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf434 = buf433; del buf433  # reuse
        buf435 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_453, input_454, input_455], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf434, primals_907, primals_908, primals_909, primals_910, primals_911, buf435, 8192, grid=grid(8192), stream=stream0)
        del primals_907
        del primals_911
        # Topologically Sorted Source Nodes: [input_456], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf435, primals_912, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf437 = buf436; del buf436  # reuse
        buf438 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_456, input_457, input_458], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf437, primals_913, primals_914, primals_915, primals_916, primals_917, buf438, 8192, grid=grid(8192), stream=stream0)
        del primals_913
        del primals_917
        # Topologically Sorted Source Nodes: [input_459], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, primals_918, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf440 = buf422; del buf422  # reuse
        buf450 = buf414; del buf414  # reuse
        # Topologically Sorted Source Nodes: [input_459, low3_14, input_460, input_461], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_native_batch_norm_backward_relu_40.run(buf439, primals_919, buf431, primals_920, primals_921, primals_922, primals_923, buf440, buf450, 16384, grid=grid(16384), stream=stream0)
        del primals_920
        del primals_923
        # Topologically Sorted Source Nodes: [input_462], Original ATen: [aten.convolution]
        buf441 = extern_kernels.convolution(buf440, primals_924, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf441, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf442 = buf441; del buf441  # reuse
        buf443 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_462, input_463, input_464], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf442, primals_925, primals_926, primals_927, primals_928, primals_929, buf443, 8192, grid=grid(8192), stream=stream0)
        del primals_925
        del primals_929
        # Topologically Sorted Source Nodes: [input_465], Original ATen: [aten.convolution]
        buf444 = extern_kernels.convolution(buf443, primals_930, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf444, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf445 = buf444; del buf444  # reuse
        buf446 = empty_strided_cuda((4, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_465, input_466, input_467], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf445, primals_931, primals_932, primals_933, primals_934, primals_935, buf446, 8192, grid=grid(8192), stream=stream0)
        del primals_931
        del primals_935
        # Topologically Sorted Source Nodes: [input_468], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, primals_936, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf448 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2_3], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_41.run(buf448, 64, grid=grid(64), stream=stream0)
        buf449 = buf33; del buf33  # reuse
        buf495 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_36, up1_3, input_459, low3_14, input_468, low3_15, up2_3, add_55], Original ATen: [aten.convolution, aten.add, aten._unsafe_index, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_native_batch_norm_backward_42.run(buf449, primals_73, buf25, buf448, buf447, primals_937, buf439, primals_919, buf431, primals_56, buf495, 65536, grid=grid(65536), stream=stream0)
        del buf25
        del buf431
        del buf439
        del primals_56
        del primals_73
        del primals_919
        del primals_937
        buf458 = buf369; del buf369  # reuse
        buf459 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [input_387, low3_6], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_43.run(buf458, primals_775, buf361, primals_776, primals_758, buf459, 1024, grid=grid(1024), stream=stream0)
        del primals_758
        del primals_775
        del primals_776
        buf460 = buf352; del buf352  # reuse
        buf461 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [input_369, low3_4], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_43.run(buf460, primals_739, buf344, primals_740, primals_722, buf461, 1024, grid=grid(1024), stream=stream0)
        del primals_722
        del primals_739
        del primals_740
        buf463 = buf325; del buf325  # reuse
        buf464 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [input_342, low3_1], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_44.run(buf463, primals_685, buf317, primals_686, primals_668, buf464, 256, grid=grid(256), stream=stream0)
        del primals_668
        del primals_685
        del primals_686
        buf465 = buf308; del buf308  # reuse
        buf466 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [input_324, low2_3], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_44.run(buf465, primals_649, buf300, primals_650, primals_632, buf466, 256, grid=grid(256), stream=stream0)
        del primals_632
        del primals_649
        del primals_650
        buf467 = buf291; del buf291  # reuse
        buf468 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [input_306, low2_1], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_44.run(buf467, primals_613, buf283, primals_614, primals_596, buf468, 256, grid=grid(256), stream=stream0)
        del primals_596
        del primals_613
        del primals_614
        buf469 = buf274; del buf274  # reuse
        buf470 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [input_288, low1_19], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_44.run(buf469, primals_577, buf266, primals_578, primals_560, buf470, 256, grid=grid(256), stream=stream0)
        del primals_560
        del primals_577
        del primals_578
        buf471 = buf257; del buf257  # reuse
        buf472 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [input_270, low1_17], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_44.run(buf471, primals_541, buf249, primals_542, primals_524, buf472, 256, grid=grid(256), stream=stream0)
        del primals_524
        del primals_541
        del primals_542
        buf473 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [low1_15], Original ATen: [aten.max_pool2d_with_indices, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_45.run(buf206, primals_506, buf473, 256, grid=grid(256), stream=stream0)
        del primals_506
        buf474 = buf231; del buf231  # reuse
        buf475 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [input_243, up1_14], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_43.run(buf474, primals_487, buf223, primals_488, primals_470, buf475, 1024, grid=grid(1024), stream=stream0)
        del primals_470
        del primals_487
        del primals_488
        buf476 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [input_225, up1_12], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_46.run(buf476, primals_451, buf206, primals_452, 1024, grid=grid(1024), stream=stream0)
        del primals_451
        del primals_452
        buf478 = buf188; del buf188  # reuse
        buf479 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [input_198, low1_12], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_43.run(buf478, primals_397, buf180, primals_398, primals_380, buf479, 1024, grid=grid(1024), stream=stream0)
        del primals_380
        del primals_397
        del primals_398
        buf480 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [low1_10], Original ATen: [aten.max_pool2d_with_indices, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_47.run(buf137, primals_362, buf480, 1024, grid=grid(1024), stream=stream0)
        del primals_362
        buf481 = buf162; del buf162  # reuse
        buf482 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [input_171, up1_10], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_48.run(buf481, primals_343, buf154, primals_344, primals_326, buf482, 4096, grid=grid(4096), stream=stream0)
        del primals_326
        del primals_343
        del primals_344
        buf483 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [input_153, up1_8], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_49.run(buf483, primals_307, buf137, primals_308, 4096, grid=grid(4096), stream=stream0)
        del primals_307
        del primals_308
        buf485 = buf119; del buf119  # reuse
        buf486 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [input_126, low1_7], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_48.run(buf485, primals_253, buf111, primals_254, primals_236, buf486, 4096, grid=grid(4096), stream=stream0)
        del primals_236
        del primals_253
        del primals_254
        buf487 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [low1_5], Original ATen: [aten.max_pool2d_with_indices, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_50.run(buf68, primals_218, buf487, 4096, grid=grid(4096), stream=stream0)
        del primals_218
        buf488 = buf93; del buf93  # reuse
        buf489 = buf447; del buf447  # reuse
        # Topologically Sorted Source Nodes: [input_99, up1_6], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_51.run(buf488, primals_199, buf85, primals_200, primals_182, buf489, 16384, grid=grid(16384), stream=stream0)
        del primals_182
        del primals_199
        del primals_200
        buf490 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [input_81, up1_4], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_52.run(buf490, primals_163, buf68, primals_164, 16384, grid=grid(16384), stream=stream0)
        del primals_163
        del primals_164
        buf492 = buf50; del buf50  # reuse
        buf493 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [input_54, low1_2], Original ATen: [aten.convolution, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_51.run(buf492, primals_109, buf42, primals_110, primals_92, buf493, 16384, grid=grid(16384), stream=stream0)
        del primals_109
        del primals_110
        del primals_92
        buf494 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [low1], Original ATen: [aten.max_pool2d_with_indices, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_native_batch_norm_backward_53.run(primals_1, primals_74, buf494, 16384, grid=grid(16384), stream=stream0)
        del primals_74
        buf496 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [up1, input_18, up1_1], Original ATen: [aten.add, aten.convolution, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_batch_norm_backward_54.run(buf496, primals_37, buf8, primals_1, primals_38, 65536, grid=grid(65536), stream=stream0)
        del primals_37
        del primals_38
    return (buf449, primals_1, primals_2, primals_3, primals_6, primals_8, primals_9, primals_10, primals_12, primals_14, primals_15, primals_16, primals_18, primals_20, primals_21, primals_22, primals_24, primals_26, primals_27, primals_28, primals_30, primals_32, primals_33, primals_34, primals_36, primals_39, primals_40, primals_42, primals_44, primals_45, primals_46, primals_48, primals_50, primals_51, primals_52, primals_54, primals_57, primals_58, primals_60, primals_62, primals_63, primals_64, primals_66, primals_68, primals_69, primals_70, primals_72, primals_75, primals_78, primals_80, primals_81, primals_82, primals_84, primals_86, primals_87, primals_88, primals_90, primals_93, primals_94, primals_96, primals_98, primals_99, primals_100, primals_102, primals_104, primals_105, primals_106, primals_108, primals_111, primals_112, primals_114, primals_116, primals_117, primals_118, primals_120, primals_122, primals_123, primals_124, primals_126, primals_129, primals_130, primals_132, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_144, primals_146, primals_147, primals_148, primals_150, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_162, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_180, primals_183, primals_184, primals_186, primals_188, primals_189, primals_190, primals_192, primals_194, primals_195, primals_196, primals_198, primals_201, primals_202, primals_204, primals_206, primals_207, primals_208, primals_210, primals_212, primals_213, primals_214, primals_216, primals_219, primals_220, primals_222, primals_224, primals_225, primals_226, primals_228, primals_230, primals_231, primals_232, primals_234, primals_237, primals_238, primals_240, primals_242, primals_243, primals_244, primals_246, primals_248, primals_249, primals_250, primals_252, primals_255, primals_256, primals_258, primals_260, primals_261, primals_262, primals_264, primals_266, primals_267, primals_268, primals_270, primals_273, primals_274, primals_276, primals_278, primals_279, primals_280, primals_282, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_294, primals_296, primals_297, primals_298, primals_300, primals_302, primals_303, primals_304, primals_306, primals_309, primals_310, primals_312, primals_314, primals_315, primals_316, primals_318, primals_320, primals_321, primals_322, primals_324, primals_327, primals_328, primals_330, primals_332, primals_333, primals_334, primals_336, primals_338, primals_339, primals_340, primals_342, primals_345, primals_346, primals_348, primals_350, primals_351, primals_352, primals_354, primals_356, primals_357, primals_358, primals_360, primals_363, primals_364, primals_366, primals_368, primals_369, primals_370, primals_372, primals_374, primals_375, primals_376, primals_378, primals_381, primals_382, primals_384, primals_386, primals_387, primals_388, primals_390, primals_392, primals_393, primals_394, primals_396, primals_399, primals_400, primals_402, primals_404, primals_405, primals_406, primals_408, primals_410, primals_411, primals_412, primals_414, primals_417, primals_418, primals_420, primals_422, primals_423, primals_424, primals_426, primals_428, primals_429, primals_430, primals_432, primals_434, primals_435, primals_436, primals_438, primals_440, primals_441, primals_442, primals_444, primals_446, primals_447, primals_448, primals_450, primals_453, primals_454, primals_456, primals_458, primals_459, primals_460, primals_462, primals_464, primals_465, primals_466, primals_468, primals_471, primals_472, primals_474, primals_476, primals_477, primals_478, primals_480, primals_482, primals_483, primals_484, primals_486, primals_489, primals_490, primals_492, primals_494, primals_495, primals_496, primals_498, primals_500, primals_501, primals_502, primals_504, primals_507, primals_508, primals_510, primals_512, primals_513, primals_514, primals_516, primals_518, primals_519, primals_520, primals_522, primals_525, primals_526, primals_528, primals_530, primals_531, primals_532, primals_534, primals_536, primals_537, primals_538, primals_540, primals_543, primals_544, primals_546, primals_548, primals_549, primals_550, primals_552, primals_554, primals_555, primals_556, primals_558, primals_561, primals_562, primals_564, primals_566, primals_567, primals_568, primals_570, primals_572, primals_573, primals_574, primals_576, primals_579, primals_580, primals_582, primals_584, primals_585, primals_586, primals_588, primals_590, primals_591, primals_592, primals_594, primals_597, primals_598, primals_600, primals_602, primals_603, primals_604, primals_606, primals_608, primals_609, primals_610, primals_612, primals_615, primals_616, primals_618, primals_620, primals_621, primals_622, primals_624, primals_626, primals_627, primals_628, primals_630, primals_633, primals_634, primals_636, primals_638, primals_639, primals_640, primals_642, primals_644, primals_645, primals_646, primals_648, primals_651, primals_652, primals_654, primals_656, primals_657, primals_658, primals_660, primals_662, primals_663, primals_664, primals_666, primals_669, primals_670, primals_672, primals_674, primals_675, primals_676, primals_678, primals_680, primals_681, primals_682, primals_684, primals_687, primals_688, primals_690, primals_692, primals_693, primals_694, primals_696, primals_698, primals_699, primals_700, primals_702, primals_705, primals_706, primals_708, primals_710, primals_711, primals_712, primals_714, primals_716, primals_717, primals_718, primals_720, primals_723, primals_724, primals_726, primals_728, primals_729, primals_730, primals_732, primals_734, primals_735, primals_736, primals_738, primals_741, primals_742, primals_744, primals_746, primals_747, primals_748, primals_750, primals_752, primals_753, primals_754, primals_756, primals_759, primals_760, primals_762, primals_764, primals_765, primals_766, primals_768, primals_770, primals_771, primals_772, primals_774, primals_777, primals_778, primals_780, primals_782, primals_783, primals_784, primals_786, primals_788, primals_789, primals_790, primals_792, primals_795, primals_796, primals_798, primals_800, primals_801, primals_802, primals_804, primals_806, primals_807, primals_808, primals_810, primals_813, primals_814, primals_816, primals_818, primals_819, primals_820, primals_822, primals_824, primals_825, primals_826, primals_828, primals_831, primals_832, primals_834, primals_836, primals_837, primals_838, primals_840, primals_842, primals_843, primals_844, primals_846, primals_849, primals_850, primals_852, primals_854, primals_855, primals_856, primals_858, primals_860, primals_861, primals_862, primals_864, primals_867, primals_868, primals_870, primals_872, primals_873, primals_874, primals_876, primals_878, primals_879, primals_880, primals_882, primals_885, primals_886, primals_888, primals_890, primals_891, primals_892, primals_894, primals_896, primals_897, primals_898, primals_900, primals_903, primals_904, primals_906, primals_908, primals_909, primals_910, primals_912, primals_914, primals_915, primals_916, primals_918, primals_921, primals_922, primals_924, primals_926, primals_927, primals_928, primals_930, primals_932, primals_933, primals_934, primals_936, buf0, buf2, buf3, buf5, buf6, buf8, buf9, buf11, buf12, buf14, buf15, buf17, buf19, buf20, buf22, buf23, buf26, buf28, buf29, buf31, buf32, buf34, buf36, buf37, buf39, buf40, buf43, buf45, buf46, buf48, buf49, buf51, buf53, buf54, buf56, buf57, buf60, buf62, buf63, buf65, buf66, buf68, buf69, buf71, buf72, buf74, buf75, buf77, buf79, buf80, buf82, buf83, buf86, buf88, buf89, buf91, buf92, buf94, buf96, buf97, buf99, buf100, buf102, buf103, buf105, buf106, buf108, buf109, buf112, buf114, buf115, buf117, buf118, buf120, buf122, buf123, buf125, buf126, buf129, buf131, buf132, buf134, buf135, buf137, buf138, buf140, buf141, buf143, buf144, buf146, buf148, buf149, buf151, buf152, buf155, buf157, buf158, buf160, buf161, buf163, buf165, buf166, buf168, buf169, buf171, buf172, buf174, buf175, buf177, buf178, buf181, buf183, buf184, buf186, buf187, buf189, buf191, buf192, buf194, buf195, buf198, buf200, buf201, buf203, buf204, buf206, buf207, buf209, buf210, buf212, buf213, buf215, buf217, buf218, buf220, buf221, buf224, buf226, buf227, buf229, buf230, buf232, buf234, buf235, buf237, buf238, buf240, buf241, buf243, buf244, buf246, buf247, buf250, buf252, buf253, buf255, buf256, buf258, buf260, buf261, buf263, buf264, buf267, buf269, buf270, buf272, buf273, buf275, buf277, buf278, buf280, buf281, buf284, buf286, buf287, buf289, buf290, buf292, buf294, buf295, buf297, buf298, buf301, buf303, buf304, buf306, buf307, buf309, buf311, buf312, buf314, buf315, buf318, buf320, buf321, buf323, buf324, buf326, buf328, buf329, buf331, buf332, buf335, buf337, buf338, buf340, buf341, buf343, buf345, buf347, buf348, buf350, buf351, buf353, buf355, buf356, buf358, buf359, buf362, buf364, buf365, buf367, buf368, buf370, buf372, buf373, buf375, buf376, buf378, buf380, buf382, buf383, buf385, buf386, buf388, buf390, buf391, buf393, buf394, buf397, buf399, buf400, buf402, buf403, buf405, buf407, buf408, buf410, buf411, buf413, buf415, buf417, buf418, buf420, buf421, buf423, buf425, buf426, buf428, buf429, buf432, buf434, buf435, buf437, buf438, buf440, buf442, buf443, buf445, buf446, buf448, buf450, buf451, buf452, buf453, buf454, buf455, buf456, buf457, buf458, buf459, buf460, buf461, buf462, buf463, buf464, buf465, buf466, buf467, buf468, buf469, buf470, buf471, buf472, buf473, buf474, buf475, buf476, buf477, buf478, buf479, buf480, buf481, buf482, buf483, buf484, buf485, buf486, buf487, buf488, buf489, buf490, buf491, buf492, buf493, buf494, buf495, buf496, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_924 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_927 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_930 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_933 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((4, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
