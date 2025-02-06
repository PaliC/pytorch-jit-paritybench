# AOT ID: ['9_inference']
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


# kernel path: inductor_cache/l6/cl6zunbgcqqnykzhzmvkjbxwhobwzpl34cgvhf6wvq5nzw5a37hd.py
# Topologically Sorted Source Nodes: [max_1, normalized_activations_step_0], Original ATen: [aten.max, aten.sub]
# Source node to ATen node mapping:
#   max_1 => max_1
#   normalized_activations_step_0 => sub
# Graph fragment:
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%arg0_1, -1, True), kwargs = {})
#   %sub : [num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %getitem), kwargs = {})
triton_poi_fused_max_sub_0 = async_compile.triton('triton_poi_fused_max_sub_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7n/c7n3u5pjw7rf555tc3xdehexjhexzfiyorukypw2zi5tddurrvpy.py
# Topologically Sorted Source Nodes: [mul, add, relu, pow_1, logt_partition, pow_2, normalized_activations], Original ATen: [aten.mul, aten.add, aten.relu, aten.pow, aten.sum]
# Source node to ATen node mapping:
#   add => add
#   logt_partition => sum_1
#   mul => mul
#   normalized_activations => mul_1
#   pow_1 => pow_1
#   pow_2 => pow_2
#   relu => relu
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -3.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 1.0), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu, -0.3333333333333333), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [-1], True), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, -3.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %pow_2), kwargs = {})
triton_poi_fused_add_mul_pow_relu_sum_1 = async_compile.triton('triton_poi_fused_add_mul_pow_relu_sum_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_relu_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_relu_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = -3.0
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = -0.3333333333333333
    tmp9 = libdevice.pow(tmp7, tmp8)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = triton_helpers.maximum(tmp6, tmp12)
    tmp14 = libdevice.pow(tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp16 * tmp2
    tmp18 = tmp17 + tmp4
    tmp19 = triton_helpers.maximum(tmp6, tmp18)
    tmp20 = libdevice.pow(tmp19, tmp8)
    tmp21 = tmp15 + tmp20
    tmp23 = tmp22 * tmp2
    tmp24 = tmp23 + tmp4
    tmp25 = triton_helpers.maximum(tmp6, tmp24)
    tmp26 = libdevice.pow(tmp25, tmp8)
    tmp27 = tmp21 + tmp26
    tmp28 = tl.full([1], 1, tl.int32)
    tmp29 = tmp28 / tmp27
    tmp30 = tmp29 * tmp29
    tmp31 = tmp30 * tmp29
    tmp32 = tmp0 * tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ix/cixpvkp6g3zltsr5gggmflwb2fbchfkpez4iv6dygp2obyxvd7wy.py
# Topologically Sorted Source Nodes: [mul_2, add_1, relu_1, pow_3, logt_partition_1, pow_4, normalized_activations_1], Original ATen: [aten.mul, aten.add, aten.relu, aten.pow, aten.sum]
# Source node to ATen node mapping:
#   add_1 => add_1
#   logt_partition_1 => sum_2
#   mul_2 => mul_2
#   normalized_activations_1 => mul_3
#   pow_3 => pow_3
#   pow_4 => pow_4
#   relu_1 => relu_1
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, -3.0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 1.0), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_1, -0.3333333333333333), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [-1], True), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, -3.0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %pow_4), kwargs = {})
triton_poi_fused_add_mul_pow_relu_sum_2 = async_compile.triton('triton_poi_fused_add_mul_pow_relu_sum_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_relu_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_relu_sum_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr1 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = -3.0
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = -0.3333333333333333
    tmp9 = libdevice.pow(tmp7, tmp8)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = triton_helpers.maximum(tmp6, tmp12)
    tmp14 = libdevice.pow(tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp16 * tmp2
    tmp18 = tmp17 + tmp4
    tmp19 = triton_helpers.maximum(tmp6, tmp18)
    tmp20 = libdevice.pow(tmp19, tmp8)
    tmp21 = tmp15 + tmp20
    tmp23 = tmp22 * tmp2
    tmp24 = tmp23 + tmp4
    tmp25 = triton_helpers.maximum(tmp6, tmp24)
    tmp26 = libdevice.pow(tmp25, tmp8)
    tmp27 = tmp21 + tmp26
    tmp28 = tl.full([1], 1, tl.int32)
    tmp29 = tmp28 / tmp27
    tmp30 = tmp29 * tmp29
    tmp31 = tmp30 * tmp29
    tmp32 = tmp0 * tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r2/cr2w3dfiobjvkp5yxb4qam5xkwcwyeaujb5gijmoegyibhdqg5gz.py
# Topologically Sorted Source Nodes: [mul_8, add_4, relu_4, pow_9, logt_partition_4, pow_10, normalized_activations_4], Original ATen: [aten.mul, aten.add, aten.relu, aten.pow, aten.sum]
# Source node to ATen node mapping:
#   add_4 => add_4
#   logt_partition_4 => sum_5
#   mul_8 => mul_8
#   normalized_activations_4 => mul_9
#   pow_10 => pow_10
#   pow_9 => pow_9
#   relu_4 => relu_4
# Graph fragment:
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, -3.0), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, 1.0), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_4, -0.3333333333333333), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [-1], True), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_5, -3.0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %pow_10), kwargs = {})
triton_poi_fused_add_mul_pow_relu_sum_3 = async_compile.triton('triton_poi_fused_add_mul_pow_relu_sum_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_relu_sum_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_relu_sum_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = -3.0
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = -0.3333333333333333
    tmp9 = libdevice.pow(tmp7, tmp8)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = triton_helpers.maximum(tmp6, tmp12)
    tmp14 = libdevice.pow(tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp16 * tmp2
    tmp18 = tmp17 + tmp4
    tmp19 = triton_helpers.maximum(tmp6, tmp18)
    tmp20 = libdevice.pow(tmp19, tmp8)
    tmp21 = tmp15 + tmp20
    tmp23 = tmp22 * tmp2
    tmp24 = tmp23 + tmp4
    tmp25 = triton_helpers.maximum(tmp6, tmp24)
    tmp26 = libdevice.pow(tmp25, tmp8)
    tmp27 = tmp21 + tmp26
    tmp28 = tl.full([1], 1, tl.int32)
    tmp29 = tmp28 / tmp27
    tmp30 = tmp29 * tmp29
    tmp31 = tmp30 * tmp29
    tmp32 = tmp0 * tmp31
    tl.store(in_out_ptr0 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/76/c76hbzufsdjwy3iuzxaqybv6z2sw6i56frxg6d4kuaxlpbbp3pbt.py
# Topologically Sorted Source Nodes: [max_1, mul_10, add_5, relu_5, pow_11, logt_partition_5, truediv, pow_12, sub_1, truediv_1, neg, normalization_constants], Original ATen: [aten.max, aten.mul, aten.add, aten.relu, aten.pow, aten.sum, aten.reciprocal, aten.sub, aten.div, aten.neg]
# Source node to ATen node mapping:
#   add_5 => add_5
#   logt_partition_5 => sum_6
#   max_1 => max_1
#   mul_10 => mul_10
#   neg => neg
#   normalization_constants => add_6
#   pow_11 => pow_11
#   pow_12 => pow_12
#   relu_5 => relu_5
#   sub_1 => sub_1
#   truediv => mul_11, reciprocal
#   truediv_1 => div
# Graph fragment:
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%arg0_1, -1, True), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, -3.0), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, 1.0), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_5, -0.3333333333333333), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [-1], True), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sum_6,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_11, -3.0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_12, 1.0), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, -3.0), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %getitem), kwargs = {})
triton_poi_fused_add_div_max_mul_neg_pow_reciprocal_relu_sub_sum_4 = async_compile.triton('triton_poi_fused_add_div_max_mul_neg_pow_reciprocal_relu_sub_sum_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_max_mul_neg_pow_reciprocal_relu_sub_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_max_mul_neg_pow_reciprocal_relu_sub_sum_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = -0.3333333333333333
    tmp8 = libdevice.pow(tmp6, tmp7)
    tmp10 = tmp9 * tmp1
    tmp11 = tmp10 + tmp3
    tmp12 = triton_helpers.maximum(tmp5, tmp11)
    tmp13 = libdevice.pow(tmp12, tmp7)
    tmp14 = tmp8 + tmp13
    tmp16 = tmp15 * tmp1
    tmp17 = tmp16 + tmp3
    tmp18 = triton_helpers.maximum(tmp5, tmp17)
    tmp19 = libdevice.pow(tmp18, tmp7)
    tmp20 = tmp14 + tmp19
    tmp22 = tmp21 * tmp1
    tmp23 = tmp22 + tmp3
    tmp24 = triton_helpers.maximum(tmp5, tmp23)
    tmp25 = libdevice.pow(tmp24, tmp7)
    tmp26 = tmp20 + tmp25
    tmp27 = tl.full([1], 1, tl.int32)
    tmp28 = tmp27 / tmp26
    tmp29 = tmp28 * tmp3
    tmp30 = tmp27 / tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tmp31 * tmp30
    tmp33 = tmp32 - tmp3
    tmp34 = tmp33 * tmp7
    tmp35 = -tmp34
    tmp38 = triton_helpers.maximum(tmp36, tmp37)
    tmp40 = triton_helpers.maximum(tmp38, tmp39)
    tmp42 = triton_helpers.maximum(tmp40, tmp41)
    tmp43 = tmp35 + tmp42
    tl.store(in_out_ptr0 + (x0), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/42/c42b6fgn5zgfcitcd3slaha2el6ov3qyuoao2nmlvvkghyhzbjd4.py
# Topologically Sorted Source Nodes: [max_1, add_8, pow_14, sub_3, truediv_2, mul_12, sub_1, truediv_1, neg, normalization_constants, sub_2, mul_11, add_7, relu_6, probabilities, pow_15, sub_4, truediv_3, mul_13, sub_5, pow_16, truediv_4, sub_6], Original ATen: [aten.max, aten.add, aten.pow, aten.sub, aten.div, aten.mul, aten.neg, aten.relu]
# Source node to ATen node mapping:
#   add_7 => add_7
#   add_8 => add_8
#   max_1 => max_1
#   mul_11 => mul_12
#   mul_12 => mul_13
#   mul_13 => mul_14
#   neg => neg
#   normalization_constants => add_6
#   pow_14 => pow_14
#   pow_15 => pow_15
#   pow_16 => pow_16
#   probabilities => pow_13
#   relu_6 => relu_6
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_6 => sub_6
#   truediv_1 => div
#   truediv_2 => div_1
#   truediv_3 => div_2
#   truediv_4 => div_3
# Graph fragment:
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%arg0_1, -1, True), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, 1e-10), kwargs = {})
#   %pow_14 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, -3.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_14, 1.0), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, -3.0), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %div_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_12, 1.0), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, -3.0), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %getitem), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %add_6), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, -3.0), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, 1.0), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
#   %pow_13 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_6, -0.3333333333333333), kwargs = {})
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%pow_13, -3.0), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_15, 1.0), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_4, -3.0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %div_2), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %pow_16 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, -2.0), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_16, -2.0), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_5, %div_3), kwargs = {})
triton_poi_fused_add_div_max_mul_neg_pow_relu_sub_5 = async_compile.triton('triton_poi_fused_add_div_max_mul_neg_pow_relu_sub_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_max_mul_neg_pow_relu_sub_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_max_mul_neg_pow_relu_sub_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp12 = tl.load(in_ptr1 + (x2), xmask)
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 1e-10
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp3 / tmp2
    tmp5 = tmp4 * tmp4
    tmp6 = tmp5 * tmp4
    tmp7 = 1.0
    tmp8 = tmp6 - tmp7
    tmp9 = -0.3333333333333333
    tmp10 = tmp8 * tmp9
    tmp11 = tmp0 * tmp10
    tmp14 = tmp12 - tmp13
    tmp15 = -3.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16 + tmp7
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = libdevice.pow(tmp19, tmp9)
    tmp21 = tmp3 / tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tmp22 * tmp21
    tmp24 = tmp23 - tmp7
    tmp25 = tmp24 * tmp9
    tmp26 = tmp0 * tmp25
    tmp27 = tmp11 - tmp26
    tmp28 = tmp3 / tmp0
    tmp29 = tmp28 * tmp28
    tmp30 = -0.5
    tmp31 = tmp29 * tmp30
    tmp32 = tmp27 - tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ui/cuicjynvsdm6r42dwguajflxpgycuqdctuf6ecaurjbrzfurkou5.py
# Topologically Sorted Source Nodes: [max_1, sub_1, truediv_1, neg, normalization_constants, sub_2, mul_11, add_7, relu_6, probabilities, pow_17, truediv_5, loss_values, loss_values_1, loss], Original ATen: [aten.max, aten.sub, aten.div, aten.neg, aten.add, aten.mul, aten.relu, aten.pow, aten.sum, aten.mean]
# Source node to ATen node mapping:
#   add_7 => add_7
#   loss => mean
#   loss_values => add_9
#   loss_values_1 => sum_7
#   max_1 => max_1
#   mul_11 => mul_12
#   neg => neg
#   normalization_constants => add_6
#   pow_17 => pow_17
#   probabilities => pow_13
#   relu_6 => relu_6
#   sub_1 => sub_1
#   sub_2 => sub_2
#   truediv_1 => div
#   truediv_5 => div_4
# Graph fragment:
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%arg0_1, -1, True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_12, 1.0), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, -3.0), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %getitem), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %add_6), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, -3.0), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, 1.0), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
#   %pow_13 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_6, -0.3333333333333333), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%pow_13, -2.0), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_17, -2.0), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_6, %div_4), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_9, [-1]), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_7,), kwargs = {})
triton_per_fused_add_div_max_mean_mul_neg_pow_relu_sub_sum_6 = async_compile.triton('triton_per_fused_add_div_max_mean_mul_neg_pow_relu_sub_sum_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_max_mean_mul_neg_pow_relu_sub_sum_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_max_mean_mul_neg_pow_relu_sub_sum_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_out_ptr0 + (r0), None)
    tmp18 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = -3.0
    tmp5 = tmp3 * tmp4
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1, 1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = -0.3333333333333333
    tmp11 = libdevice.pow(tmp9, tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = -0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tmp0 + tmp16
    tmp20 = tmp19 - tmp2
    tmp21 = tmp20 * tmp4
    tmp22 = tmp21 + tmp6
    tmp23 = triton_helpers.maximum(tmp8, tmp22)
    tmp24 = libdevice.pow(tmp23, tmp10)
    tmp25 = tmp12 / tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tmp26 * tmp15
    tmp28 = tmp18 + tmp27
    tmp29 = tmp17 + tmp28
    tmp32 = tmp31 - tmp2
    tmp33 = tmp32 * tmp4
    tmp34 = tmp33 + tmp6
    tmp35 = triton_helpers.maximum(tmp8, tmp34)
    tmp36 = libdevice.pow(tmp35, tmp10)
    tmp37 = tmp12 / tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tmp38 * tmp15
    tmp40 = tmp30 + tmp39
    tmp41 = tmp29 + tmp40
    tmp44 = tmp43 - tmp2
    tmp45 = tmp44 * tmp4
    tmp46 = tmp45 + tmp6
    tmp47 = triton_helpers.maximum(tmp8, tmp46)
    tmp48 = libdevice.pow(tmp47, tmp10)
    tmp49 = tmp12 / tmp48
    tmp50 = tmp49 * tmp49
    tmp51 = tmp50 * tmp15
    tmp52 = tmp42 + tmp51
    tmp53 = tmp41 + tmp52
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
    tmp56 = tl.sum(tmp54, 1)[:, None]
    tmp57 = 64.0
    tmp58 = tmp56 / tmp57
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp58, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_1, normalized_activations_step_0], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_0.run(arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, add, relu, pow_1, logt_partition, pow_2, normalized_activations], Original ATen: [aten.mul, aten.add, aten.relu, aten.pow, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_relu_sum_1.run(buf0, buf1, 256, grid=grid(256), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_2, add_1, relu_1, pow_3, logt_partition_1, pow_4, normalized_activations_1], Original ATen: [aten.mul, aten.add, aten.relu, aten.pow, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_relu_sum_2.run(buf0, buf1, buf2, 256, grid=grid(256), stream=stream0)
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [mul_4, add_2, relu_2, pow_5, logt_partition_2, pow_6, normalized_activations_2], Original ATen: [aten.mul, aten.add, aten.relu, aten.pow, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_relu_sum_2.run(buf0, buf2, buf3, 256, grid=grid(256), stream=stream0)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [mul_6, add_3, relu_3, pow_7, logt_partition_3, pow_8, normalized_activations_3], Original ATen: [aten.mul, aten.add, aten.relu, aten.pow, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_relu_sum_2.run(buf0, buf3, buf4, 256, grid=grid(256), stream=stream0)
        del buf3
        buf5 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul_8, add_4, relu_4, pow_9, logt_partition_4, pow_10, normalized_activations_4], Original ATen: [aten.mul, aten.add, aten.relu, aten.pow, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_relu_sum_3.run(buf5, buf4, 256, grid=grid(256), stream=stream0)
        del buf4
        buf6 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [max_1, mul_10, add_5, relu_5, pow_11, logt_partition_5, truediv, pow_12, sub_1, truediv_1, neg, normalization_constants], Original ATen: [aten.max, aten.mul, aten.add, aten.relu, aten.pow, aten.sum, aten.reciprocal, aten.sub, aten.div, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_max_mul_neg_pow_reciprocal_relu_sub_sum_4.run(buf7, buf5, arg0_1, 64, grid=grid(64), stream=stream0)
        buf8 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [max_1, add_8, pow_14, sub_3, truediv_2, mul_12, sub_1, truediv_1, neg, normalization_constants, sub_2, mul_11, add_7, relu_6, probabilities, pow_15, sub_4, truediv_3, mul_13, sub_5, pow_16, truediv_4, sub_6], Original ATen: [aten.max, aten.add, aten.pow, aten.sub, aten.div, aten.mul, aten.neg, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_max_mul_neg_pow_relu_sub_5.run(arg1_1, arg0_1, buf7, buf8, 256, grid=grid(256), stream=stream0)
        del arg1_1
        buf9 = reinterpret_tensor(buf7, (4, 4, 4), (16, 4, 1), 0); del buf7  # reuse
        buf10 = empty_strided_cuda((), (), torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [max_1, sub_1, truediv_1, neg, normalization_constants, sub_2, mul_11, add_7, relu_6, probabilities, pow_17, truediv_5, loss_values, loss_values_1, loss], Original ATen: [aten.max, aten.sub, aten.div, aten.neg, aten.add, aten.mul, aten.relu, aten.pow, aten.sum, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_max_mean_mul_neg_pow_relu_sub_sum_6.run(buf9, buf11, buf8, arg0_1, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del buf8
        del buf9
    return (buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
