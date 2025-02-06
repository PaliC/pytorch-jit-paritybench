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


# kernel path: inductor_cache/q3/cq3qwsdllbfe4kcnzafrkbigmr6zqgewbe5hu6wk6mahffnatncn.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => gt, mul_3, where
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/zw/czwvwgpdjcpr3sw7ydfpm3lg27r6lmwxwgskmxcsj7um3kjonf4i.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_5 => add_3, mul_5, mul_6, sub_1
#   input_6 => gt_1, mul_7, where_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.1), kwargs = {})
#   %where_1 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_3, %mul_7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/er/cereulram4k5ybfkkd3kilveoow3a2rvotbwaria3f6vw4th7asg.py
# Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_8 => add_5, mul_10, mul_9, sub_2
#   input_9 => gt_2, mul_11, where_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_5, 0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.1), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_5, %mul_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/2g/c2gfaumjx446r6a6lxt72u6pbcvg6mnx4q6go6sshgvgijaqd7dd.py
# Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_14 => add_9, mul_17, mul_18, sub_4
#   input_15 => gt_4, mul_19, where_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 0.1), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %add_9, %mul_19), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 512)
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/yz/cyzh3xheflalchgla3gld7jiilduavcczf3mq4u3w255hgarxpqx.py
# Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_20 => add_13, mul_25, mul_26, sub_6
#   input_21 => gt_6, mul_27, where_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_55), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_13, 0), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, 0.1), kwargs = {})
#   %where_6 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_13, %mul_27), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/w5/cw5hosjsseqfgy5yizvpptkzzr7gbvaiktn4275yjshouughvbzl.py
# Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_26 => add_17, mul_33, mul_34, sub_8
#   input_27 => gt_8, mul_35, where_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %unsqueeze_69), kwargs = {})
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %unsqueeze_71), kwargs = {})
#   %gt_8 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_17, 0), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, 0.1), kwargs = {})
#   %where_8 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %add_17, %mul_35), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/dk/cdkszpedvfflpj2cat4jrxyhczgnzbccydtlkavaplpcqatn7oky.py
# Topologically Sorted Source Nodes: [concat5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concat5 => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_7, %where_10, %convolution_11], 1), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 1026)
    x0 = (xindex % 4)
    x2 = xindex // 4104
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 2048*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 1024, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*((-512) + x1) + 2048*x2), tmp9 & xmask, other=0.0)
    tmp11 = 0.0
    tmp12 = tmp10 > tmp11
    tmp13 = 0.1
    tmp14 = tmp10 * tmp13
    tmp15 = tl.where(tmp12, tmp10, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp9, tmp15, tmp16)
    tmp18 = tmp0 >= tmp7
    tmp19 = tl.full([1], 1026, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr2 + (x0 + 4*((-1024) + x1) + 8*x2), tmp18 & xmask, other=0.0)
    tmp22 = tl.where(tmp9, tmp17, tmp21)
    tmp23 = tl.where(tmp4, tmp5, tmp22)
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6k/c6k32z2vzbfo6u7obzpbicpusoo7uj3lj56ksibyk2gyj3zrvy4d.py
# Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   input_32 => gt_10, mul_40, where_10
# Graph fragment:
#   %gt_10 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_12, 0), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_12, 0.1), kwargs = {})
#   %where_10 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_10, %convolution_12, %mul_40), kwargs = {})
#   %gt_17 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_10, 0), kwargs = {})
triton_poi_fused_leaky_relu_leaky_relu_backward_7 = async_compile.triton('triton_poi_fused_leaky_relu_leaky_relu_backward_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/ck/ccksl5jkm6qcy2oxuwdypmcm72evzvhtn47w7et6vkgvbav6llrk.py
# Topologically Sorted Source Nodes: [concat4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concat4 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_5, %where_11, %convolution_14], 1), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 770)
    x0 = (xindex % 16)
    x2 = xindex // 12320
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 8192*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 768, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-512) + x1) + 4096*x2), tmp9 & xmask, other=0.0)
    tmp11 = 0.0
    tmp12 = tmp10 > tmp11
    tmp13 = 0.1
    tmp14 = tmp10 * tmp13
    tmp15 = tl.where(tmp12, tmp10, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp9, tmp15, tmp16)
    tmp18 = tmp0 >= tmp7
    tmp19 = tl.full([1], 770, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr2 + (x0 + 16*((-768) + x1) + 32*x2), tmp18 & xmask, other=0.0)
    tmp22 = tl.where(tmp9, tmp17, tmp21)
    tmp23 = tl.where(tmp4, tmp5, tmp22)
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ou/couh2sxmd7hrdemcqrrcbdhvzscqvfynw6ddz2ikzdfpzjfppc44.py
# Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   input_34 => gt_11, mul_41, where_11
# Graph fragment:
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_15, 0), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, 0.1), kwargs = {})
#   %where_11 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_11, %convolution_15, %mul_41), kwargs = {})
#   %gt_16 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_11, 0), kwargs = {})
triton_poi_fused_leaky_relu_leaky_relu_backward_9 = async_compile.triton('triton_poi_fused_leaky_relu_leaky_relu_backward_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/76/c766ynwdooe7lms6yay37bynynngsvp6vykt3i5g5yhnmbjut7sj.py
# Topologically Sorted Source Nodes: [concat3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concat3 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_3, %where_12, %convolution_17], 1), kwargs = {})
triton_poi_fused_cat_10 = async_compile.triton('triton_poi_fused_cat_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 386)
    x0 = (xindex % 64)
    x2 = xindex // 24704
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 16384*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 384, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 64*((-256) + x1) + 8192*x2), tmp9 & xmask, other=0.0)
    tmp11 = 0.0
    tmp12 = tmp10 > tmp11
    tmp13 = 0.1
    tmp14 = tmp10 * tmp13
    tmp15 = tl.where(tmp12, tmp10, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp9, tmp15, tmp16)
    tmp18 = tmp0 >= tmp7
    tmp19 = tl.full([1], 386, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr2 + (x0 + 64*((-384) + x1) + 128*x2), tmp18 & xmask, other=0.0)
    tmp22 = tl.where(tmp9, tmp17, tmp21)
    tmp23 = tl.where(tmp4, tmp5, tmp22)
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pk/cpkhzdr6y2r7dujuuzjjgqygexmcdhnd4rmtdxnjwumhrs4bpg6u.py
# Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   input_36 => gt_12, mul_42, where_12
# Graph fragment:
#   %gt_12 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_18, 0), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_18, 0.1), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %convolution_18, %mul_42), kwargs = {})
#   %gt_15 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_12, 0), kwargs = {})
triton_poi_fused_leaky_relu_leaky_relu_backward_11 = async_compile.triton('triton_poi_fused_leaky_relu_leaky_relu_backward_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/ml/cmlsz2rc45hlnlqh3xvhiurymg6etkmmpp4537bt4cgs5i2rcjqz.py
# Topologically Sorted Source Nodes: [concat2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concat2 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%where_1, %where_13, %convolution_20], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 198656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 256) % 194)
    x0 = (xindex % 256)
    x2 = xindex // 49664
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 32768*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-128) + x1) + 16384*x2), tmp9 & xmask, other=0.0)
    tmp11 = 0.0
    tmp12 = tmp10 > tmp11
    tmp13 = 0.1
    tmp14 = tmp10 * tmp13
    tmp15 = tl.where(tmp12, tmp10, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp9, tmp15, tmp16)
    tmp18 = tmp0 >= tmp7
    tmp19 = tl.full([1], 194, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr2 + (x0 + 256*((-192) + x1) + 512*x2), tmp18 & xmask, other=0.0)
    tmp22 = tl.where(tmp9, tmp17, tmp21)
    tmp23 = tl.where(tmp4, tmp5, tmp22)
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ua/cuasu6pvmkbhvjuil3etlygvd7m2i4oatosork66o2w2rpudzuvc.py
# Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   input_38 => gt_13, mul_43, where_13
# Graph fragment:
#   %gt_13 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_21, 0), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_21, 0.1), kwargs = {})
#   %where_13 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_13, %convolution_21, %mul_43), kwargs = {})
#   %gt_14 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_13, 0), kwargs = {})
triton_poi_fused_leaky_relu_leaky_relu_backward_13 = async_compile.triton('triton_poi_fused_leaky_relu_leaky_relu_backward_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64 = args
    args.clear()
    assert_size_stride(primals_1, (64, 6, 7, 7), (294, 49, 7, 1))
    assert_size_stride(primals_2, (4, 6, 64, 64), (24576, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (128, 64, 5, 5), (1600, 25, 5, 1))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (256, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_43, (1024, ), (1, ))
    assert_size_stride(primals_44, (1024, ), (1, ))
    assert_size_stride(primals_45, (1024, ), (1, ))
    assert_size_stride(primals_46, (1024, ), (1, ))
    assert_size_stride(primals_47, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_48, (1024, ), (1, ))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_50, (1024, ), (1, ))
    assert_size_stride(primals_51, (1024, ), (1, ))
    assert_size_stride(primals_52, (2, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_53, (2, 2, 4, 4), (32, 16, 4, 1))
    assert_size_stride(primals_54, (1024, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(primals_55, (2, 1026, 3, 3), (9234, 9, 3, 1))
    assert_size_stride(primals_56, (2, 2, 4, 4), (32, 16, 4, 1))
    assert_size_stride(primals_57, (1026, 256, 4, 4), (4096, 16, 4, 1))
    assert_size_stride(primals_58, (2, 770, 3, 3), (6930, 9, 3, 1))
    assert_size_stride(primals_59, (2, 2, 4, 4), (32, 16, 4, 1))
    assert_size_stride(primals_60, (770, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_61, (2, 386, 3, 3), (3474, 9, 3, 1))
    assert_size_stride(primals_62, (2, 2, 4, 4), (32, 16, 4, 1))
    assert_size_stride(primals_63, (386, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_64, (2, 194, 3, 3), (1746, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_0.run(buf2, buf0, primals_3, primals_4, primals_5, primals_6, 262144, grid=grid(262144), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_7, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf4 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_1.run(buf5, buf3, primals_8, primals_9, primals_10, primals_11, 131072, grid=grid(131072), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_12, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf7 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2.run(buf8, buf6, primals_13, primals_14, primals_15, primals_16, 65536, grid=grid(65536), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf10 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2.run(buf11, buf9, primals_18, primals_19, primals_20, primals_21, 65536, grid=grid(65536), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf13 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_3.run(buf14, buf12, primals_23, primals_24, primals_25, primals_26, 32768, grid=grid(32768), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf16 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_3.run(buf17, buf15, primals_28, primals_29, primals_30, primals_31, 32768, grid=grid(32768), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_32, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf19 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4.run(buf20, buf18, primals_33, primals_34, primals_35, primals_36, 8192, grid=grid(8192), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf22 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4.run(buf23, buf21, primals_38, primals_39, primals_40, primals_41, 8192, grid=grid(8192), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_42, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 1024, 1, 1), (1024, 1, 1, 1))
        buf25 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf26 = reinterpret_tensor(buf25, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5.run(buf26, buf24, primals_43, primals_44, primals_45, primals_46, 4096, grid=grid(4096), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 1024, 1, 1), (1024, 1, 1, 1))
        buf28 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf29 = reinterpret_tensor(buf28, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5.run(buf29, buf27, primals_48, primals_49, primals_50, primals_51, 4096, grid=grid(4096), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [flow6], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 2, 1, 1), (2, 1, 1, 1))
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf29, primals_54, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 512, 2, 2), (2048, 4, 2, 1))
        # Topologically Sorted Source Nodes: [flow6_up], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_53, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 2, 2, 2), (8, 4, 2, 1))
        buf33 = empty_strided_cuda((4, 1026, 2, 2), (4104, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf23, buf32, buf31, buf33, 16416, grid=grid(16416), stream=stream0)
        del buf31
        buf50 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_7.run(buf32, buf50, 8192, grid=grid(8192), stream=stream0)
        del buf32
        # Topologically Sorted Source Nodes: [flow5], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 2, 2, 2), (8, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf33, primals_57, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 256, 4, 4), (4096, 16, 4, 1))
        # Topologically Sorted Source Nodes: [flow5_up], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_56, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 2, 4, 4), (32, 16, 4, 1))
        buf37 = empty_strided_cuda((4, 770, 4, 4), (12320, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf17, buf36, buf35, buf37, 49280, grid=grid(49280), stream=stream0)
        del buf35
        buf49 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_9.run(buf36, buf49, 16384, grid=grid(16384), stream=stream0)
        del buf36
        # Topologically Sorted Source Nodes: [flow4], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 2, 4, 4), (32, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf37, primals_60, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 128, 8, 8), (8192, 64, 8, 1))
        # Topologically Sorted Source Nodes: [flow4_up], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_59, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 2, 8, 8), (128, 64, 8, 1))
        buf41 = empty_strided_cuda((4, 386, 8, 8), (24704, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf11, buf40, buf39, buf41, 98816, grid=grid(98816), stream=stream0)
        del buf39
        buf48 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_11.run(buf40, buf48, 32768, grid=grid(32768), stream=stream0)
        del buf40
        # Topologically Sorted Source Nodes: [flow3], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 2, 8, 8), (128, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf41, primals_63, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 64, 16, 16), (16384, 256, 16, 1))
        # Topologically Sorted Source Nodes: [flow3_up], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_62, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 2, 16, 16), (512, 256, 16, 1))
        buf45 = empty_strided_cuda((4, 194, 16, 16), (49664, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf5, buf44, buf43, buf45, 198656, grid=grid(198656), stream=stream0)
        del buf43
        buf47 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_13.run(buf44, buf47, 65536, grid=grid(65536), stream=stream0)
        del buf44
        # Topologically Sorted Source Nodes: [flow2], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 2, 16, 16), (512, 256, 16, 1))
    return (buf46, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, buf0, buf2, buf3, buf5, buf6, buf8, buf9, buf11, buf12, buf14, buf15, buf17, buf18, buf20, buf21, buf23, buf24, buf26, buf27, buf29, buf30, buf33, buf34, buf37, buf38, buf41, buf42, buf45, buf47, buf48, buf49, buf50, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 6, 7, 7), (294, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 6, 64, 64), (24576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 64, 5, 5), (1600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((2, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((2, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1024, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((2, 1026, 3, 3), (9234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((2, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1026, 256, 4, 4), (4096, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((2, 770, 3, 3), (6930, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((2, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((770, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((2, 386, 3, 3), (3474, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((2, 2, 4, 4), (32, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((386, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((2, 194, 3, 3), (1746, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
